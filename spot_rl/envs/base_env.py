import time

import cv2
import gym
import magnum as mn
import numpy as np
import quaternion
import rospy
from mask_rcnn_detectron2.inference import MaskRcnnInference
from sensor_msgs.msg import CompressedImage
from spot_wrapper.spot import Spot, wrap_heading
from spot_wrapper.utils import say

from spot_rl.spot_ros_node import (
    MASK_RCNN_VIZ_TOPIC,
    MAX_DEPTH,
    MAX_GRIPPER_DEPTH,
    SpotRosSubscriber,
)


def pad_action(action):
    """We only control 4 out of 6 joints; add zeros to non-controllable indices."""
    return np.array([*action[:3], 0.0, action[3], 0.0])


def rescale_actions(actions, action_thresh=0.05):
    actions = np.clip(actions, -1, 1)
    # Silence low actions
    actions[np.abs(actions) < action_thresh] = 0.0
    # Remap action scaling to compensate for silenced values
    action_offsets = []
    for v in actions:
        if v > 0:
            action_offsets.append(action_thresh)
        elif v < 0:
            action_offsets.append(-action_thresh)
        else:
            action_offsets.append(0)
    actions = (actions - np.array(action_offsets)) / (1.0 - action_thresh)

    return actions


class SpotBaseEnv(SpotRosSubscriber, gym.Env):
    def __init__(
        self, config, spot: Spot, mask_rcnn_weights=None, mask_rcnn_device=None
    ):
        super().__init__("spot_reality_gym")
        self.config = config
        self.spot = spot

        # General environment parameters
        self.ctrl_hz = config.CTRL_HZ
        self.max_episode_steps = config.MAX_EPISODE_STEPS
        self.last_execution = time.time()
        self.should_end = False
        self.num_steps = 0
        self.reset_ran = False

        # Base action parameters
        self.max_lin_vel = config.MAX_LIN_VEL
        self.max_ang_vel = config.MAX_ANG_VEL

        # Arm action parameters
        self.initial_arm_joint_angles = np.deg2rad(config.INITIAL_ARM_JOINT_ANGLES)
        self.max_ang_vel = config.MAX_ANG_VEL
        self.actually_move_arm = config.ACTUALLY_MOVE_ARM
        self.arm_lower_limits = np.deg2rad(config.ARM_LOWER_LIMITS)
        self.arm_upper_limits = np.deg2rad(config.ARM_UPPER_LIMITS)
        self.locked_on_object_count = 0
        self.grasp_attempted = False
        self.place_attempted = False

        # Mask RCNN
        if mask_rcnn_weights is not None:
            self.mrcnn = MaskRcnnInference(
                mask_rcnn_weights, score_thresh=0.5, device=mask_rcnn_device
            )
        else:
            self.mrcnn = None
        self.mrcnn_viz_pub = rospy.Publisher(
            MASK_RCNN_VIZ_TOPIC, CompressedImage, queue_size=1
        )
        self.obj_center_pixel = None

        # Arrange Spot into initial configuration
        assert spot.spot_lease is not None, "Need motor control of Spot!"

    def viz_callback(self, msg):
        # This node does not process mrcnn visualizations
        pass

    def robot_state_callback(self, msg):
        # Transform robot's xy_yaw to be in home frame
        super().robot_state_callback(msg)
        self.x, self.y, self.yaw = self.spot.xy_yaw_global_to_home(
            self.x, self.y, self.yaw
        )

    def reset(self, *args, **kwargs):
        # Reset parameters
        self.num_steps = 0
        self.reset_ran = True
        self.should_end = False
        self.grasp_attempted = False
        self.place_attempted = False

        self.decompress_imgs()
        observations = self.get_observations()
        return observations

    def step(
        self,
        base_action=None,
        arm_action=None,
        grasp=False,
        place=False,
        max_joint_movement_key="MAX_JOINT_MOVEMENT",
    ):
        """Moves the arm and returns updated observations

        :param base_action: np.array of velocities (lineaer, angular)
        :param arm_action: np.array of radians denoting how each joint is to be moved
        :param grasp: whether to call the grasp_hand_depth() METHOD
        :param place: whether to call the open_gripper() method
        :param max_joint_movement_key: max allowable displacement of arm joints
        :return:
        """
        assert self.reset_ran, ".reset() must be called first!"
        assert base_action is not None or arm_action is not None, "Must provide action."
        if grasp:
            print("GRASP ACTION CALLED: Grasping center object!")
            # The following cmd is blocking
            self.spot.grasp_hand_depth()

            # Just leave the object on the receptacle if desired
            if self.config.DONT_PICK_UP:
                self.spot.open_gripper()

            # Return to pre-grasp joint positions after grasp
            cmd_id = self.spot.set_arm_joint_positions(
                positions=self.initial_arm_joint_angles, travel_time=1.0
            )
            self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)
            self.grasp_attempted = True
        elif place:
            print("PLACE ACTION CALLED: Opening the gripper!")
            self.spot.open_gripper()
            self.place_attempted = True
        else:
            if base_action is not None:
                # Command velocities using the input action
                x_vel, ang_vel = base_action
                x_vel = np.clip(x_vel, -1, 1) * self.max_lin_vel
                ang_vel = np.clip(ang_vel, -1, 1) * self.max_ang_vel
                # No horizontal velocity
                self.spot.set_base_velocity(x_vel, 0.0, ang_vel, 1 / self.ctrl_hz)

            if arm_action is not None:
                arm_action = rescale_actions(arm_action)
                # Insert zeros for joints we don't control
                padded_action = np.clip(pad_action(arm_action), -1.0, 1.0)
                scaled_action = padded_action * self.config[max_joint_movement_key]
                target_positions = self.current_arm_pose + scaled_action
                target_positions = np.clip(
                    target_positions, self.arm_lower_limits, self.arm_upper_limits
                )
                if self.actually_move_arm:
                    _ = self.spot.set_arm_joint_positions(
                        positions=target_positions, travel_time=1 / self.ctrl_hz * 0.9
                    )
        # Pause until enough time has passed during this step
        while time.time() < self.last_execution + 1 / self.ctrl_hz:
            pass
        env_hz = 1 / (time.time() - self.last_execution)
        self.last_execution = time.time()

        self.decompress_imgs()
        observations = self.get_observations()

        self.num_steps += 1
        timeout = self.num_steps == self.max_episode_steps
        done = timeout or self.get_success(observations) or self.should_end

        # Don't need reward or info
        reward = None
        info = {"env_hz": env_hz}

        return observations, reward, done, info

    @staticmethod
    def get_nav_success(observations, success_distance, success_angle):
        # Is the agent at the goal?
        dist_to_goal, _ = observations["target_point_goal_gps_and_compass_sensor"]
        at_goal = dist_to_goal < success_distance
        good_heading = abs(observations["goal_heading"][0]) < success_angle
        return at_goal and good_heading

    def print_nav_stats(self, observations):
        rho, theta = observations["target_point_goal_gps_and_compass_sensor"]
        print(
            f"Dist to goal: {rho:.2f}\t"
            f"theta: {np.rad2deg(theta):.2f}\t"
            f"x: {self.x:.2f}\t"
            f"y: {self.y:.2f}\t"
            f"yaw: {np.rad2deg(self.yaw):.2f}\t"
            f"gh: {np.rad2deg(observations['goal_heading'][0]):.2f}\t"
        )

    def get_nav_observation(self, goal_xy, goal_heading):
        observations = {}

        # Get visual observations
        front_depth = self.filter_depth(
            self.front_depth, max_depth=MAX_DEPTH, whiten_black=True
        )

        front_depth = cv2.resize(
            front_depth, (120 * 2, 212), interpolation=cv2.INTER_AREA
        )
        front_depth = np.float32(front_depth) / 255.0
        # Add dimension for channel (unsqueeze)
        front_depth = front_depth.reshape(*front_depth.shape[:2], 1)
        observations["spot_right_depth"], observations["spot_left_depth"] = np.split(
            front_depth, 2, 1
        )

        # Get rho theta observation
        curr_xy = np.array([self.x, self.y], dtype=np.float32)
        rho = np.linalg.norm(curr_xy - goal_xy)
        theta = np.arctan2(goal_xy[1] - self.y, goal_xy[0] - self.x) - self.yaw
        rho_theta = np.array([rho, wrap_heading(theta)], dtype=np.float32)

        # Get goal heading observation
        goal_heading_ = -np.array([goal_heading - self.yaw], dtype=np.float32)
        observations["target_point_goal_gps_and_compass_sensor"] = rho_theta
        observations["goal_heading"] = goal_heading_

        return observations

    def get_arm_joints(self):
        # Get proprioception inputs
        joints = np.array(
            [
                j
                for idx, j in enumerate(self.current_arm_pose)
                if idx not in self.config.JOINT_BLACKLIST
            ],
            dtype=np.float32,
        )

        return joints

    def get_gripper_images(
        self, target_obj_id, left_crop=124, right_crop=60, new_width=228, new_height=240
    ):
        arm_depth = self.hand_depth.copy()
        orig_height, orig_width = arm_depth.shape[:2]
        obj_bbox = self.get_mrcnn_det(target_obj_id)

        # Crop out black vertical bars on the left and right edges of aligned depth img
        arm_depth = arm_depth[:, left_crop:-right_crop]

        # Blur and resize
        arm_depth = self.filter_depth(arm_depth, max_depth=MAX_GRIPPER_DEPTH)
        arm_depth = cv2.resize(
            arm_depth, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        arm_depth = arm_depth.reshape([*arm_depth.shape, 1])  # unsqueeze

        # Generate object mask channel
        arm_depth_bbox = np.zeros_like(arm_depth)
        if obj_bbox is not None:
            x1, y1, x2, y2 = obj_bbox
            width_scale = new_width / orig_width
            height_scale = new_height / orig_height
            x1 = int((x1 - left_crop) * width_scale)
            x2 = int((x2 - left_crop) * width_scale)
            y1 = int(y1 * height_scale)
            y2 = int(y2 * height_scale)
            arm_depth_bbox[y1:y2, x1:x2] = 1.0

        # Normalize
        arm_depth = np.float32(arm_depth) / 255.0

        return arm_depth, arm_depth_bbox

    def get_mrcnn_det(self, target_obj_id):
        img = cv2.cvtColor(self.hand_rgb, cv2.COLOR_BGR2RGB)
        pred = self.mrcnn.inference(img)

        if len(pred["instances"]) > 0:
            det_str = self.format_detections(pred["instances"])
            viz_img = self.mrcnn.visualize_inference(img, pred)
            img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(viz_img)
            self.mrcnn_viz_pub.publish(img_msg)
            print("[mask_rcnn]: " + det_str)
        else:
            det_str = "None"

        if det_str == "None":
            return None

        # Check if desired object is in view of camera
        def correct_class(detection):
            return int(detection.split(",")[0]) == target_obj_id

        matching_detections = [d for d in det_str.split(";") if correct_class(d)]
        if not matching_detections:
            return None

        # Get object match with the highest score
        def get_score(detection):
            return float(detection.split(",")[1])

        best_detection = sorted(matching_detections, key=get_score)[-1]
        x1, y1, x2, y2 = [int(float(i)) for i in best_detection.split(",")[-4:]]

        # Create bbox mask from selected detection
        height, width = img.shape[:2]
        cx = np.mean([x1, x2])
        cy = np.mean([y1, y2])
        self.obj_center_pixel = (cx, cy)

        locked_on = self.locked_on_object(x1, y1, x2, y2, height, width)
        if locked_on:
            self.locked_on_object_count += 1
            self.spot.loginfo(
                f"Locked on to target {self.locked_on_object_count} time(s)..."
            )
        else:
            if self.locked_on_object_count > 0:
                self.spot.loginfo("Lost lock-on!")
            self.locked_on_object_count = 0

        return x1, y1, x2, y2

    @staticmethod
    def locked_on_object(x1, y1, x2, y2, height, width, radius=0.1):
        cy, cx = height // 2, width // 2
        # Locked on if the center of the image is in the bbox
        if x1 < cx < x2 and y1 < cy < y2:
            return True

        pixel_radius = min(height, width) * radius
        # Get pixel distance between bbox rectangle and the center of the image
        # Stack Overflow question ID #5254838
        dx = np.max([x1 - cx, 0, cx - x2])
        dy = np.max([y1 - cy, 0, cy - y2])
        bbox_dist = np.sqrt(dx ** 2 + dy ** 2)
        locked_on = bbox_dist < pixel_radius

        return locked_on

    @staticmethod
    def format_detections(detections):
        detection_str = []
        for det_idx in range(len(detections)):
            class_id = detections.pred_classes[det_idx]
            score = detections.scores[det_idx]
            x1, y1, x2, y2 = detections.pred_boxes[det_idx].tensor.squeeze(0)
            det_attrs = [str(i.item()) for i in [class_id, score, x1, y1, x2, y2]]
            detection_str.append(",".join(det_attrs))
        detection_str = ";".join(detection_str)
        return detection_str

    def get_observations(self):
        raise NotImplementedError

    def get_success(self, observations):
        raise NotImplementedError

    def should_end(self):
        return False

    @staticmethod
    def spot2habitat_transform(position, rotation):
        x, y, z = position.x, position.y, position.z
        qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w

        quat = quaternion.quaternion(qw, qx, qy, qz)
        rotation_matrix = mn.Quaternion(quat.imag, quat.real).to_matrix()
        rotation_matrix_fixed = (
            rotation_matrix
            @ mn.Matrix4.rotation(
                mn.Rad(-np.pi / 2.0), mn.Vector3(1.0, 0.0, 0.0)
            ).rotation()
        )
        translation = mn.Vector3(x, z, -y)

        quat_rotated = mn.Quaternion.from_matrix(rotation_matrix_fixed)
        quat_rotated.vector = mn.Vector3(
            quat_rotated.vector[0], quat_rotated.vector[2], -quat_rotated.vector[1]
        )
        rotation_matrix_fixed = quat_rotated.to_matrix()
        sim_transform = mn.Matrix4.from_(rotation_matrix_fixed, translation)

        return sim_transform

    @staticmethod
    def spot2habitat_translation(spot_translation):
        return mn.Vector3(np.array(spot_translation)[np.array([0, 2, 1])])

    @property
    def curr_transform(self):
        # Assume body is at default height of 0.5 m
        # This is local_T_global.
        return mn.Matrix4.from_(
            mn.Matrix4.rotation_z(mn.Rad(self.yaw)).rotation(),
            mn.Vector3(self.x, self.y, 0.5),
        )

    def power_robot(self):
        self.spot.power_on()
        say("Standing up")
        self.spot.blocking_stand()
