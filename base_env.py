import time

import cv2
import gym
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    image_response_to_cv2,
    scale_depth_img,
    wrap_heading,
)
from spot_wrapper.utils import say
from std_msgs.msg import String

SPOT_IMAGE_SOURCES = [
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME,
    SpotCamIds.HAND_COLOR,
]

FRONT_DEPTH_TOPIC = "/spot_cams/filtered_front_depth"
HAND_DEPTH_TOPIC = f"/spot_cams/{SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME}"
DET_TOPIC = "/mask_rcnn_detections"

CTRL_HZ = 4
MAX_EPISODE_STEPS = 100


# Base action params
MAX_LIN_VEL = 0.5  # m/s
MAX_ANG_VEL = 0.523599  # 30 degrees/s, in radians
VEL_TIME = 0.5

# Arm action params
MAX_JOINT_MOVEMENT = 0.0698132
INITIAL_ARM_JOINT_ANGLES = np.deg2rad([0, -170, 120, 0, 75, 0])
ARM_LOWER_LIMITS = np.deg2rad([-45, -180, 0, 0, -90, 0])
ARM_UPPER_LIMITS = np.deg2rad([45, -45, 135, 0, 90, 0])
JOINT_BLACKLIST = ["arm0.el0", "arm0.wr1"]  # joints we can't control
ACTUALLY_MOVE_ARM = True
CENTER_TOLERANCE = 0.25


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


class SpotBaseEnv(gym.Env):
    def __init__(self, spot: Spot):
        self.spot = spot

        # ROS subscribers
        rospy.init_node("spot_reality_gym", disable_signals=True)  # enable Ctrl + C
        rospy.Subscriber(FRONT_DEPTH_TOPIC, Image, self.front_depth_cb)
        rospy.Subscriber(HAND_DEPTH_TOPIC, Image, self.hand_depth_cb)
        rospy.Subscriber(DET_TOPIC, String, self.detection_cb)
        self.cv_bridge = CvBridge()
        self.front_depth_img = None
        self.hand_depth_img = None
        self.detections = "None"

        # General environment parameters
        self.ctrl_hz = CTRL_HZ
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.last_execution = time.time()
        self.should_end = False
        self.num_steps = 0
        self.reset_ran = False

        # Robot state parameters
        self.x, self.y, self.yaw = None, None, None
        self.current_arm_pose = None

        # Base action parameters
        self.max_lin_vel = MAX_LIN_VEL
        self.max_ang_vel = MAX_ANG_VEL
        self.vel_time = VEL_TIME

        # Arm action parameters
        self.initial_arm_joint_angles = INITIAL_ARM_JOINT_ANGLES
        self.max_joint_movement = MAX_JOINT_MOVEMENT
        self.max_ang_vel = MAX_ANG_VEL
        self.vel_time = VEL_TIME
        self.actually_move_arm = ACTUALLY_MOVE_ARM
        self.arm_lower_limits = ARM_LOWER_LIMITS
        self.arm_upper_limits = ARM_UPPER_LIMITS
        self.locked_on_object_count = 0

        # Arrange Spot into initial configuration
        assert spot.spot_lease is not None, "Need motor control of Spot!"
        spot.power_on()
        say("Standing up")
        spot.blocking_stand()

    def front_depth_cb(self, msg):
        self.front_depth_img = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")

    def hand_depth_cb(self, msg):
        self.hand_depth_img = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")

    def detection_cb(self, str_msg):
        self.detections = str_msg.data

    def reset(self, *args, **kwargs):
        # Reset parameters
        self.num_steps = 0
        self.reset_ran = True
        self.should_end = False

        # Save latest position of each arm joint
        self.get_arm_joints()

        observations = self.get_observations()
        return observations

    def step(self, base_action=None, arm_action=None, grasp=False, place=False):
        """Moves the arm and returns updated observations

        :param base_action: np.array of velocities (lineaer, angular)
        :param arm_action: np.array of radians denoting how each joint is to be moved
        :param grasp: whether to call the grasp_center_of_hand_depth() METHOD
        :param place: whether to call the open_gripper() method
        :return:
        """
        assert self.reset_ran, ".reset() must be called first!"
        assert base_action is not None or arm_action is not None, "Must provide action."

        if grasp:
            print("GRASP ACTION CALLED: Grasping center object!")
            # The following cmd is blocking
            self.spot.grasp_center_of_hand_depth()
            # Return to pre-grasp joint positions after grasp
            cmd_id = self.spot.set_arm_joint_positions(
                positions=self.initial_arm_joint_angles, travel_time=1.0
            )
            self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)
        elif place:
            print("PLACE ACTION CALLED: Opening the gripper!")
            self.spot.open_gripper()
        else:
            if base_action is not None:
                # Command velocities using the input action
                x_vel, ang_vel = base_action
                x_vel = np.clip(x_vel, -1, 1) * self.max_lin_vel
                ang_vel = np.clip(ang_vel, -1, 1) * self.max_ang_vel
                # No horizontal velocity
                self.spot.set_base_velocity(x_vel, 0.0, ang_vel, self.vel_time)

            if arm_action is not None:
                arm_action = rescale_actions(arm_action)
                # Insert zeros for joints we don't control
                padded_action = np.clip(pad_action(arm_action), -1.0, 1.0)
                scaled_action = padded_action * self.max_joint_movement
                target_positions = self.current_arm_pose + scaled_action
                target_positions = np.clip(
                    target_positions, self.arm_lower_limits, self.arm_upper_limits
                )
                if self.actually_move_arm:
                    _ = self.spot.set_arm_joint_positions(
                        positions=target_positions, travel_time=1 / self.ctrl_hz * 1.1
                    )

        # Pause until enough time has passed during this step
        while time.time() < self.last_execution + 1 / self.ctrl_hz:
            pass
        print("Env Hz:", 1 / (time.time() - self.last_execution))
        self.last_execution = time.time()

        observations = self.get_observations()

        self.num_steps += 1
        timeout = self.num_steps == self.max_episode_steps
        done = timeout or self.get_success(observations) or self.should_end

        # Don't need reward or info
        reward, info = None, {}
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
        front_depth = cv2.resize(
            self.front_depth_img, (120 * 2, 212), interpolation=cv2.INTER_AREA
        )
        front_depth = np.float32(front_depth) / 255.0
        # Add dimension for channel (unsqueeze)
        front_depth = front_depth.reshape(*front_depth.shape[:2], 1)
        observations["spot_left_depth"], observations["spot_right_depth"] = np.split(
            front_depth, 2, 1
        )

        # Get rho theta observation
        self.x, self.y, self.yaw = self.spot.get_xy_yaw()
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
        arm_proprioception = self.spot.get_arm_proprioception().values()
        self.current_arm_pose = np.array([j.position.value for j in arm_proprioception])
        joints = np.array(
            [
                j.position.value
                for j in arm_proprioception
                if j.name not in JOINT_BLACKLIST
            ],
            dtype=np.float32,
        )

        return joints

    def get_gripper_images(self, target_obj_id):
        image_responses = self.spot.get_image_responses(
            [SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
        )
        raw_arm_depth = image_response_to_cv2(image_responses[0])
        arm_depth = scale_depth_img(raw_arm_depth, max_depth=1.7)
        arm_depth = cv2.resize(arm_depth, (320, 240))

        # arm_depth = cv2.resize(self.hand_depth_img, (320, 240))

        arm_depth = arm_depth.reshape([*arm_depth.shape, 1])  # unsqueeze
        arm_depth_bbox = self.get_mrcnn_det(target_obj_id)

        return arm_depth, arm_depth_bbox

    def get_mrcnn_det(self, target_obj_id):
        arm_depth_bbox = np.zeros([240, 320, 1], dtype=np.float32)
        if self.detections == "None":
            return arm_depth_bbox

        # Check if desired object is in view of camera
        def correct_class(detection):
            return int(detection.split(",")[0]) == target_obj_id

        matching_detections = [
            d for d in self.detections.split(";") if correct_class(d)
        ]
        if not matching_detections:
            return arm_depth_bbox

        # Get object match with the highest score
        def get_score(detection):
            return float(detection.split(",")[1])

        best_detection = sorted(matching_detections, key=get_score)[-1]
        x1, y1, x2, y2 = [int(float(i)) for i in best_detection.split(",")[-4:]]

        # Create bbox mask from selected detection
        # TODO: Make this less ugly
        height, width = 480.0, 640.0
        cx = np.mean([x1, x2]) / width
        cy = np.mean([y1, y2]) / height
        y_min, y_max = int(y1 / 2), int(y2 / 2)
        x_min, x_max = int(x1 / 2), int(x2 / 2)
        arm_depth_bbox[y_min:y_max, x_min:x_max] = 1.0

        # Determine if bbox intersects with central crosshair
        crosshair_in_bbox = x1 < width // 2 < x2 and y1 < height // 2 < y2

        locked_on = crosshair_in_bbox and all(
            [abs(c - 0.5) < CENTER_TOLERANCE for c in [cx, cy]]
        )
        if locked_on:
            self.locked_on_object_count += 1
            self.spot.loginfo(
                f"Locked on to target {self.locked_on_object_count} time(s)..."
            )
        else:
            if self.locked_on_object_count > 0:
                self.spot.loginfo("Lost lock-on!")
            self.locked_on_object_count = 0

        return arm_depth_bbox

    def get_observations(self):
        raise NotImplementedError

    def get_success(self, observations):
        raise NotImplementedError

    def should_end(self):
        return False
