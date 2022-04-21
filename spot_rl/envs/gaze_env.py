import cv2
import numpy as np
from spot_wrapper.spot import Spot

from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import GazePolicy
from spot_rl.utils.utils import construct_config, get_default_parser

DEBUG = False


def main(spot):
    parser = get_default_parser()
    args = parser.parse_args()
    config = construct_config(args.opts)

    env = SpotGazeEnv(config, spot, mask_rcnn_weights=config.WEIGHTS.MRCNN)
    env.power_robot()
    policy = GazePolicy(config.WEIGHTS.GAZE, device=config.DEVICE)
    policy.reset()
    observations = env.reset()
    done = False
    env.say("Starting episode")
    try:
        while not done:
            action = policy.act(observations)
            observations, _, done, _ = env.step(arm_action=action)
    finally:
        spot.power_off()


class SpotGazeEnv(SpotBaseEnv):
    def reset(self, target_obj_id=None, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=1
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=1)
        self.spot.open_gripper()

        observations = super().reset(target_obj_id=target_obj_id, *args, **kwargs)

        # Reset parameters
        self.locked_on_object_count = 0
        if target_obj_id is None:
            self.target_obj_name = self.config.TARGET_OBJ_NAME

        return observations

    def step(self, base_action=None, arm_action=None, grasp=False, place=False):
        if self.locked_on_object_count == self.config.OBJECT_LOCK_ON_NEEDED:
            grasp = True

        observations, reward, done, info = super().step(
            base_action, arm_action, grasp, place
        )

        return observations, reward, done, info

    def get_observations(self):
        arm_depth, arm_depth_bbox = self.get_gripper_images()
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
        return self.grasp_attempted


if __name__ == "__main__":
    spot = Spot("RealGazeEnv")
    with spot.get_lease(hijack=True):
        main(spot)
