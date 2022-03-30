import time

import magnum as mn
import numpy as np
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

from base_env import SpotBaseEnv
from gaze_env import SpotGazeEnv
from place_env import SpotPlaceEnv
from real_policy import GazePolicy, NavPolicy, PlacePolicy

TARGET_OBJ_ID = 3
SUCCESS_DISTANCE = 0.2
SUCCESS_ANGLE_DIST = np.deg2rad(5)
OBJECT_LOCK_ON_NEEDED = 2
PLACES_NEEDED = 3
PLACE_TARGET = mn.Vector3(1.03, -0.22, 0.37)
GAZE_DESTINATION = [
    (4.558099926809142, 1.9298394486256936),
    np.deg2rad(90.0),
]
PLACE_DESTINATION = [
    (3.5124963425520868, -1.8734770435783519),
    np.deg2rad(-87.77534964347153),
]

NAV_WEIGHTS = "weights/CUTOUT_WT_True_SD_200_ckpt.99.pvp.pth"
GAZE_WEIGHTS = "weights/speed_seed1_speed0.0872665_1648513272.ckpt.19.pth"
PLACE_WEIGHTS = "weights/rel_place_energy_manual_seed10_ckpt.49.pth"
MASK_RCNN_WEIGHTS = "weights/model_0007499.pth"

# DEVICE = "cpu"
DEVICE = "cuda:0"


def main(spot):
    policy = SequentialExperts(NAV_WEIGHTS, GAZE_WEIGHTS, PLACE_WEIGHTS, device=DEVICE)
    policy.reset()

    env = SpotMobileManipulationSeqEnv(
        spot, mask_rcnn_weights=MASK_RCNN_WEIGHTS, mask_rcnn_device=DEVICE
    )
    observations = env.reset()
    done = False
    say("Starting episode")
    time.sleep(2)
    expert = Tasks.NAV
    try:
        while not done:
            base_action, arm_action = policy.act(observations, expert=expert)
            observations, _, done, info = env.step(
                base_action=base_action, arm_action=arm_action
            )
            expert = info["correct_skill"]

            # Print info
            stats = [f"{k}: {v}" for k, v in info.items()]
            print("\t".join(stats))

            # We reuse nav, so we have to reset it before we use it again.
            if expert != Tasks.NAV:
                policy.nav_policy.reset()

        say("Environment is done.")
        time.sleep(20)
    finally:
        spot.power_off()


class Tasks:
    r"""Enumeration of types of tasks."""

    NAV = "nav"
    GAZE = "gaze"
    PLACE = "place"


class SequentialExperts:
    def __init__(self, nav_weights, gaze_weights, place_weights, device="cuda"):
        print("Loading nav_policy...")
        self.nav_policy = NavPolicy(nav_weights, device)
        print("Loading gaze_policy...")
        self.gaze_policy = GazePolicy(gaze_weights, device)
        print("Loading place_policy...")
        self.place_policy = PlacePolicy(place_weights, device)
        print("Done loading all policies!")

    def reset(self):
        self.nav_policy.reset()
        self.gaze_policy.reset()
        self.place_policy.reset()

    def act(self, observations, expert):
        base_action, arm_action = None, None
        if expert == Tasks.NAV:
            base_action = self.nav_policy.act(observations)
        elif expert == Tasks.GAZE:
            arm_action = self.gaze_policy.act(observations)
        elif expert == Tasks.PLACE:
            arm_action = self.place_policy.act(observations)

        return base_action, arm_action


class SpotMobileManipulationBaseEnv(SpotGazeEnv, SpotPlaceEnv, SpotBaseEnv):
    def __init__(self, spot: Spot, **kwargs):
        SpotBaseEnv.__init__(self, spot, **kwargs)

        # Nav
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = SUCCESS_DISTANCE
        self.succ_angle = SUCCESS_ANGLE_DIST

        # Gaze
        self.locked_on_object_count = 0
        self.target_obj_id = TARGET_OBJ_ID

        # Place
        self.place_target = PLACE_TARGET
        self.places_needed = PLACES_NEEDED
        self.place_attempts = 0
        self.prev_joints = None
        self.placed = False

        # General
        self.gaze_destination = GAZE_DESTINATION
        self.place_destination = PLACE_DESTINATION
        self.max_episode_steps = 1000

    def reset(self):
        # Move arm to initial configuration (w/ gripper open)
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=0.75
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)
        self.spot.open_gripper()

        # Nav
        self.goal_xy, self.goal_heading = self.gaze_destination
        assert len(self.goal_xy) == 2

        # Gaze
        self.locked_on_object_count = 0

        # Place
        self.placed = False

        return SpotBaseEnv.reset(self)

    def step(self, *args, **kwargs):
        return SpotBaseEnv.step(self, *args, **kwargs)

    def get_observations(self):
        observations = self.get_nav_observation(self.goal_xy, self.goal_heading)
        observations.update(super().get_observations())
        observations["obj_start_sensor"] = self.get_place_dist(
            self.place_target,
            place_nav_targ=[*PLACE_DESTINATION[0], PLACE_DESTINATION[-1]],
        )

        SpotPlaceEnv.update_place_attempts(self, observations["joint"])

        return observations


class SpotMobileManipulationSeqEnv(
    SpotMobileManipulationBaseEnv, SpotGazeEnv, SpotPlaceEnv
):
    def __init__(self, spot, **kwargs):
        super().__init__(spot, **kwargs)
        self.current_task = Tasks.NAV
        self.nav_succ_count = 0

    def reset(self):
        observations = super().reset()
        self.current_task = Tasks.NAV
        self.nav_succ_count = 0
        return observations

    def step(self, grasp=False, place=False, *args, **kwargs):
        place = (
            self.place_attempts >= self.places_needed
            and self.current_task == Tasks.PLACE
        )

        grasp = (
            self.locked_on_object_count == OBJECT_LOCK_ON_NEEDED
            and self.current_task == Tasks.GAZE
        )

        observations, reward, done, info = super().step(
            grasp=grasp, place=place, *args, **kwargs
        )

        if self.current_task == Tasks.NAV and self.get_nav_success(
            observations, self.succ_distance, self.succ_angle
        ):
            self.nav_succ_count += 1
            if self.nav_succ_count == 1:
                self.spot.set_base_velocity(0.0, 0.0, 0.0, self.vel_time)
                if not self.grasp_attempted:
                    self.current_task = Tasks.GAZE
                    self.goal_xy, self.goal_heading = self.place_destination
                else:
                    self.current_task = Tasks.PLACE
        else:
            self.nav_succ_count = 0

        if self.current_task == Tasks.GAZE and self.grasp_attempted:
            self.current_task = Tasks.NAV

        if not self.current_task == Tasks.PLACE:
            self.place_attempts = 0

        info["correct_skill"] = self.current_task
        print("correct_skill", self.current_task)

        return observations, reward, done, info

    def get_success(self, observations):
        return self.placed


if __name__ == "__main__":
    spot = Spot("RealSeqEnv")
    with spot.get_lease(hijack=True):
        main(spot)
