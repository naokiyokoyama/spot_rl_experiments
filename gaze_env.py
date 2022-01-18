import cv2
import gym
from gym import spaces
import numpy as np
import time

from gaze_policy import GazePolicy
from bd_spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    image_response_to_cv2,
    scale_depth_img,
)
from bd_spot_wrapper.utils import color_bbox, say

CTRL_HZ = 1.0
INITIAL_ARM_JOINT_ANGLES = np.deg2rad([0, -150, 120, 0, 75, 0])
JOINT_BLACKLIST = ["arm0.el0", "arm0.wr1"]  # joints we can't control
MAX_EPISODE_STEPS = 120
OBJECT_LOCK_ON_NEEDED = 3
CENTER_TOLERANCE = 0.25
ACTUALLY_GRASP = True
ACTUALLY_MOVE_ARM = True
MAX_JOINT_MOVEMENT = 0.0698132
MAX_DEPTH = 10.0

USE_MASK_RCNN = True
DET_TOPIC = "/mask_rcnn_detections"
if USE_MASK_RCNN:
    import rospy
    from std_msgs.msg import String

    TARGET_OBJ_ID = 3  # rubiks cube


def pad_action(action):
    """We only control 4 out of 6 joints; add zeros to non-controllable indices."""
    return np.array([*action[:3], 0.0, action[3], 0.0])


class SpotGazeEnv(gym.Env):
    def __init__(self, spot: Spot):
        # Standard Gym stuff
        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        # Arrange Spot into initial configuration
        assert spot.spot_lease is not None, "Need motor control of Spot!"
        spot.power_on()
        say("Standing up")
        spot.blocking_stand()
        spot.open_gripper()

        self.spot = spot
        self.locked_on_object_count = 0
        self.num_steps = 0
        self.current_positions = np.zeros(6)
        self.reset_ran = False

        # Use ROS if we use MRCNN
        if USE_MASK_RCNN:
            rospy.init_node("gaze_env")
            self.det_sub = rospy.Subscriber(DET_TOPIC, String, self.det_callback)
            self.detections = "None"
            self.target_obj_id = TARGET_OBJ_ID

    def reset(self):
        # Move arm to initial configuration
        self.spot.open_gripper()
        cmd_id = self.spot.set_arm_joint_positions(
            positions=INITIAL_ARM_JOINT_ANGLES, travel_time=2
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)

        # Reset parameters
        self.locked_on_object_count = 0
        self.num_steps = 0
        self.reset_ran = True

        observations = self.get_observations()
        return observations

    def step(self, action):
        """
        Moves the arm and returns updated observations
        :param action: np.array of radians denoting how much each joint is to be moved
        :return:
        """
        assert self.reset_ran, ".reset() must be called first!"

        # Move the arm
        padded_action = pad_action(action)  # insert zeros for joints we don't control
        padded_action = np.clip(padded_action, -1.0, 1.0) * MAX_JOINT_MOVEMENT
        target_positions = np.clip(
            self.current_positions + padded_action, -np.pi, np.pi
        )
        if ACTUALLY_MOVE_ARM:
            _ = self.spot.set_arm_joint_positions(
                positions=target_positions, travel_time=1 / CTRL_HZ
            )

        # Get observation
        time.sleep(1 / CTRL_HZ)
        observations = self.get_observations()

        # Grasp object if conditions are met
        grasp = self.locked_on_object_count == OBJECT_LOCK_ON_NEEDED
        if grasp:
            say("Locked on, executing auto grasp")
            self.spot.loginfo("Conditions for auto-grasp have been met!")
            if ACTUALLY_GRASP:
                # Grab whatever object is at the center of hand RGB camera image
                self.spot.grasp_center_of_hand_depth()
                # Return to pre-grasp joint positions
                cmd_id = self.spot.set_arm_joint_positions(
                    positions=self.current_positions, travel_time=1.0
                )
                self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)
            else:
                time.sleep(5)

        done = self.num_steps == MAX_EPISODE_STEPS or grasp

        # Don't need reward or info
        reward, info = None, {}
        return observations, reward, done, info

    def get_observations(self):
        # Get proprioception inputs
        arm_proprioception = self.spot.get_arm_proprioception()
        self.current_positions = np.array(
            [v.position.value for v in arm_proprioception.values()]
        )
        joints = np.array(
            [
                v.position.value
                for v in arm_proprioception.values()
                if v.name not in JOINT_BLACKLIST
            ],
            dtype=np.float32,
        )
        assert len(joints) == 4

        # Get visual inputs
        image_responses = self.spot.get_image_responses(
            [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
        )
        arm_rgb, raw_arm_depth = [image_response_to_cv2(i) for i in image_responses]
        arm_depth = scale_depth_img(raw_arm_depth, max_depth=MAX_DEPTH)
        arm_rgb, arm_depth = [cv2.resize(i, (320, 240)) for i in [arm_rgb, arm_depth]]
        arm_depth = arm_depth.reshape([*arm_depth.shape, 1])  # unsqueeze
        if USE_MASK_RCNN:
            arm_depth_bbox, cx, cy, crosshair_in_bbox = self.get_mrcnn_det()
        else:
            arm_depth_bbox, cx, cy, crosshair_in_bbox = color_bbox(arm_rgb)
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

        observations = {
            "joint": joints,
            "is_holding": np.zeros(1, dtype=np.float32),
            "arm_depth": arm_depth,
            "arm_depth_bbox": arm_depth_bbox,
        }

        return observations

    def get_mrcnn_det(self):
        arm_depth_bbox = np.zeros([240, 320, 1], dtype=np.float32)
        no_detection = arm_depth_bbox, 0.0, 0.0, False
        if self.detections == "None":
            return no_detection

        # Check if desired object is in view of camera
        def correct_class(detection):
            return int(detection.split(",")[0]) == self.target_obj_id
        matching_detections = [
            d for d in self.detections.split(";") if correct_class(d)
        ]
        if not matching_detections:
            return no_detection

        # Get object prediction with the highest score
        def get_score(detection):
            return float(detection.split(",")[1])
        best_detection = sorted(matching_detections, key=get_score)[-1]
        x1, y1, x2, y2 = [int(float(i)) for i in best_detection.split(",")[-4:]]

        # Create bbox mask from selected detection
        # TODO: Fix this
        height, width = 480.0, 640.0
        cx = np.mean([x1, x2]) / width
        cy = np.mean([y1, y2]) / height
        y_min, y_max = int(y1 / 2), int(y2 / 2)
        x_min, x_max = int(x1 / 2), int(x2 / 2)
        arm_depth_bbox[y_min:y_max, x_min:x_max] = 1.0

        cv2.imshow('ff', np.uint8(arm_depth_bbox * 255))
        cv2.waitKey(1)

        # Determine if bbox intersects with central crosshair
        crosshair_in_bbox = x1 < width // 2 < x2 and y1 < height // 2 < y2

        return arm_depth_bbox, cx, cy, crosshair_in_bbox

    def det_callback(self, str_msg):
        self.detections = str_msg.data


def main(spot):
    env = SpotGazeEnv(spot)
    policy = GazePolicy(
        "weights/gaze_ckpt.93.pth", device="cuda"
    )
    policy.reset()
    observations = env.reset()
    done = False
    say("Starting episode")
    time.sleep(2)
    try:
        while not done:
            # start_time = time.time()
            action = policy.act(observations)
            # print("Action inference time: ", time.time() - start_time)
            # print(action)
            observations, _, done, _ = env.step(action)
        time.sleep(20)
    finally:
        spot.power_off()

if __name__ == "__main__":
    spot = Spot("RealGazeEnv")
    with spot.get_lease():
        main(spot)
