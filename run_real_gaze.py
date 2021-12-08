from gaze_env import SpotGazeEnv
from gaze_policy import GazePolicy
from bd_spot_wrapper.utils import say
from bd_spot_wrapper.spot import Spot
import time

def main(spot):
    env = SpotGazeEnv(spot)
    policy = GazePolicy(
        "/Users/naokiyokoyama/gt/spot/spot_rl_experiments/bbox_mask_5thresh_autograsp_shortrange_seed1_36.pth",
        device="cpu",
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