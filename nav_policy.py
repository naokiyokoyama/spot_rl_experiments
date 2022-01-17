from gym import spaces
from gym.spaces import Dict as SpaceDict
import numpy as np

from gaze_policy import RealPolicy


class NavPolicy(RealPolicy):
    def __init__(self, checkpoint_path, device):
        observation_space = SpaceDict(
            {
                "spot_left_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(212, 120, 1), dtype=np.float32
                ),
                "spot_right_depth": spaces.Box(
                    low=0.0, high=1.0, shape=(212, 120, 1), dtype=np.float32
                ),
                "goal_heading": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        # Linear, angular, and horizontal velocity (in that order)
        action_space = spaces.Box(-1.0, 1.0, (2,))
        super().__init__(checkpoint_path, observation_space, action_space, device)


if __name__ == "__main__":
    nav_policy = NavPolicy(
        "weights/two_cams_with_noise_seed4_ckpt.4.pth",
        device="cpu",
    )
    nav_policy.reset()
    observations = {
        "spot_left_depth": np.zeros([212, 120, 1], dtype=np.float32),
        "spot_right_depth": np.zeros([212, 120, 1], dtype=np.float32),
        "goal_heading": np.zeros(1, dtype=np.float32),
        "pointgoal_with_gps_compass": np.zeros(2, dtype=np.float32),
    }
    actions = nav_policy.act(observations)
    print("actions:", actions)
