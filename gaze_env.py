import time

import cv2
import numpy as np
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2, scale_depth_img
from spot_wrapper.utils import say

from base_env import SpotBaseEnv
from real_policy import GazePolicy

OBJECT_LOCK_ON_NEEDED = 3
ACTUALLY_GRASP = True
ACTUALLY_MOVE_ARM = True

TARGET_OBJ_ID = 3  # rubiks cube
DEBUG = False


def main(spot):
    env = SpotGazeEnv(spot)
    print("Loading policy...")
    policy = GazePolicy("weights/real_depth_gaze_seed1_49.pth", device="cuda:0")
    print("Resetting policy")
    policy.reset()
    print("Resetting Env")
    observations = env.reset()
    for k, v in observations.items():
        print(k, v.shape)
    done = False
    say("Starting episode")
    time.sleep(2)
    try:
        while not done:
            action = policy.act(observations)
            observations, _, done, _ = env.step(arm_action=action)
        time.sleep(20)
    finally:
        spot.power_off()


class SpotGazeEnv(SpotBaseEnv):
    def __init__(self, spot: Spot):
        super().__init__(spot)

        self.locked_on_object_count = 0
        self.target_obj_id = TARGET_OBJ_ID
        self.should_end = False

    def reset(self):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=2
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)
        self.spot.open_gripper()

        observations = super().reset()

        # Reset parameters
        self.locked_on_object_count = 0
        self.target_obj_id = TARGET_OBJ_ID

        return observations

    def step(self, base_action=None, arm_action=None, grasp=False, place=False):
        if self.locked_on_object_count == OBJECT_LOCK_ON_NEEDED:
            grasp = True
            self.should_end = True

        observations, reward, done, info = super().step(
            base_action, arm_action, grasp, place
        )

        return observations, reward, done, info

    def get_observations(self):
        arm_depth, arm_depth_bbox = self.get_gripper_images(self.target_obj_id)
        if DEBUG:
            img = np.uint8(arm_depth_bbox * 255).reshape(*arm_depth_bbox.shape[:2])
            img2 = np.uint8(arm_depth * 255).reshape(*arm_depth.shape[:2])
            cv2.imwrite(f"arm_bbox_{self.num_steps:03}.png", img)
            cv2.imwrite(f"arm_depth_{self.num_steps:03}.png", img2)
        observations = {
            "joint": self.get_arm_joints(),
            "arm_depth": arm_depth,
            "arm_depth_bbox": arm_depth_bbox,
        }

        return observations

    def get_success(self, observations):
        return self.should_end


if __name__ == "__main__":
    spot = Spot("RealGazeEnv")
    with spot.get_lease(hijack=True):
        main(spot)
