import time

import magnum as mn
import numpy as np
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

from base_env import SpotBaseEnv
from real_policy import PlacePolicy

CTRL_HZ = 1.0
MAX_EPISODE_STEPS = 120
ACTUALLY_GRASP = True
ACTUALLY_MOVE_ARM = True
MAX_JOINT_MOVEMENT = 0.0698132
MAX_DEPTH = 10.0
EE_GRIPPER_OFFSET = mn.Vector3(0.2, 0.0, 0.05)
# EE_GRIPPER_OFFSET = mn.Vector3(0.0, 0.0, 0.0)
# PLACE_TARGET = mn.Vector3(1.2, 0.58, -0.55)
PLACE_TARGET = mn.Vector3(0.75, 0.25, 0.25)
PLACES_NEEDED = 3


def main(spot):
    env = SpotPlaceEnv(spot)
    policy = PlacePolicy(
        "weights/rel_place_energy_manual_seed10_ckpt.49.pth", device="cpu"
    )
    policy.reset()
    print("Loaded policy!")
    observations = env.reset()
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


class SpotPlaceEnv(SpotBaseEnv):
    def __init__(self, spot: Spot, place_target=PLACE_TARGET, target_is_local=True):
        super().__init__(spot)
        self.spot.open_gripper()
        say("Please put object in my hand.")
        time.sleep(5)
        self.spot.close_gripper()
        self.place_target = place_target
        self.target_is_local = target_is_local
        self.place_attempts = 0
        self.places_needed = PLACES_NEEDED
        self.prev_joints = None
        self.placed = False

    def reset(self, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=0.75
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)

        observations = super(SpotPlaceEnv, self).reset()
        self.placed = False
        return observations

    def step(self, place=False, *args, **kwargs):
        place = self.place_attempts >= self.places_needed
        return super().step(place=place, *args, **kwargs)

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "obj_start_sensor": self.get_place_dist(self.place_target),
        }

        self.update_place_attempts(observations["joint"])

        return observations

    def update_place_attempts(self, curr_joints):
        if self.prev_joints is None or np.any(
            np.abs(self.prev_joints - curr_joints) > np.deg2rad(1)
        ):
            self.place_attempts = 0
        else:
            self.place_attempts += 1

        self.prev_joints = curr_joints

    def get_place_dist(self, place_target):
        # The place goal should be provided relative to the local robot frame given that
        # the robot is at the place receptacle

        position, rotation = self.spot.get_base_transform_to("link_wr1")
        wrist_T_base = self.spot2habitat_transform(position, rotation)
        gripper_T_base = wrist_T_base @ mn.Matrix4.translation(EE_GRIPPER_OFFSET)
        base_T_gripper = gripper_T_base.inverted()
        gripper_pos = base_T_gripper.transform_point(place_target)

        return gripper_pos


if __name__ == "__main__":
    spot = Spot("RealPlaceEnv")
    with spot.get_lease(hijack=True):
        main(spot)
