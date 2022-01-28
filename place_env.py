import time
import magnum as mn
import numpy as np

from real_policy import PlacePolicy, spot2habitat_transform
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say
from gaze_env import SpotGazeEnv

CTRL_HZ = 1.0
MAX_EPISODE_STEPS = 120
ACTUALLY_GRASP = True
ACTUALLY_MOVE_ARM = True
MAX_JOINT_MOVEMENT = 0.0698132
MAX_DEPTH = 10.0
# EE_GRIPPER_OFFSET = mn.Vector3(0.2, 0.0, 0.05)
EE_GRIPPER_OFFSET = mn.Vector3(0.0, 0.0, 0.0)
# PLACE_TARGET = mn.Vector3(1.2, 0.58, -0.55)
PLACE_TARGET = mn.Vector3(0.75, 0.25, 0.25)

class SpotPlaceEnv(SpotGazeEnv):
    def __init__(self, spot: Spot, place_target=PLACE_TARGET):
        super().__init__(spot, use_mask_rcnn=False)
        say("Please put object in my hand.")
        time.sleep(5)
        self.spot.close_gripper()
        self.place_target = place_target
        self.place_attempts = 0
        self.prev_joints = self.get_joints()

    def final_action(self):
        attempt_place = self.place_attempts == 4
        if attempt_place:
            say("Placing object")
            self.spot.open_gripper()

        return attempt_place

    def get_place_dist(self):
        # The place goal should be provided relative to the local robot frame given that
        # the robot is at the place receptacle

        position, rotation = self.spot.get_base_transform_to("link_wr1")
        base_T_wrist = spot2habitat_transform(position, rotation)
        base_T_gripper = base_T_wrist @ mn.Matrix4.translation(EE_GRIPPER_OFFSET)
        gripper_T_base = base_T_gripper.inverted()
        gripper_pos = gripper_T_base.transform_point(self.place_target)
        print(np.linalg.norm(gripper_pos))

        return gripper_pos


    def get_observations(self):
        observations = {
            "joint": self.get_joints(),
            "obj_start_sensor": self.get_place_dist(),
        }

        if (np.abs(self.prev_joints - observations["joint"]) > np.deg2rad(1)).any():
            self.place_attempts = 0
        else:
            self.place_attempts += 1
            print("self.place_attempts", self.place_attempts)

        self.prev_joints = observations["joint"]

        return observations


def main(spot):
    env = SpotPlaceEnv(spot)
    policy = PlacePolicy(
        "weights/rel_place_energy_manual_seed10_ckpt.49.pth", device="cpu"
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
    spot = Spot("RealPlaceEnv")
    with spot.get_lease(hijack=True):
        main(spot)
