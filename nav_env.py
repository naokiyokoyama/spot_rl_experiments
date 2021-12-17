import cv2
import gym
import numpy as np
import time

from bd_spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    image_response_to_cv2,
    scale_depth_img,
)
from bd_spot_wrapper.utils import color_bbox, say

CTRL_HZ = 1.0
MAX_EPISODE_STEPS = 120

class SpotNavEnv(gym.Env):
    def __init__(self, spot: Spot):
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
        arm_depth = scale_depth_img(raw_arm_depth, max_depth=10.0)
        arm_rgb, arm_depth = [cv2.resize(i, (320, 240)) for i in [arm_rgb, arm_depth]]
        arm_depth = arm_depth.reshape([*arm_depth.shape, 1])  # unsqueeze
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
