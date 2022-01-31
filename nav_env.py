import time

import numpy as np
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

from base_env import SpotBaseEnv
from real_policy import NavPolicy

SUCCESS_DISTANCE = 0.3
SUCCESS_ANGLE_DIST = 0.0872665  # 5 radians
NAV_WEIGHTS = "weights/two_cams_with_noise_seed4_ckpt.4.pth"
GOAL_XY = [6, 0]
GOAL_HEADING = np.deg2rad(90)  # positive direction is CCW


def main(spot):
    env = SpotNavEnv(spot)
    policy = NavPolicy(NAV_WEIGHTS, device="cpu")
    policy.reset()
    observations = env.reset(GOAL_XY, GOAL_HEADING)
    done = False
    say("Starting episode")
    time.sleep(2)
    try:
        while not done:
            action = policy.act(observations)
            observations, _, done, _ = env.step(base_action=action)
        say("Environment is done.")
        time.sleep(20)
    finally:
        spot.power_off()


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, spot: Spot):
        super().__init__(spot)
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = SUCCESS_DISTANCE
        self.succ_angle = SUCCESS_ANGLE_DIST

    def reset(self, goal_xy, goal_heading):
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        observations = super().reset()
        assert len(self.goal_xy) == 2

        return observations

    def get_success(self, observations):
        return self.get_nav_success(observations, self.succ_distance, self.succ_angle)

    def get_observations(self):
        return self.get_nav_observation(self.goal_xy, self.goal_heading)


if __name__ == "__main__":
    spot = Spot("RealNavEnv")
    with spot.get_lease():
        main(spot)
