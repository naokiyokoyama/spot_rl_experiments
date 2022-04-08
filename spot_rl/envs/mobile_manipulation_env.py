import time
from collections import Counter

import magnum as mn
import numpy as np
from base_env import SpotBaseEnv
from gaze_env import SpotGazeEnv
from nav_env import SpotNavEnv
from place_env import SpotPlaceEnv
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

from spot_rl.real_policy import GazePolicy, NavPolicy, PlacePolicy
from spot_rl.utils.utils import (
    WAYPOINTS,
    closest_clutter,
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
    object_id_to_nav_waypoint,
    place_target_from_waypoints,
)

NUM_OBJECTS = 6


def main(spot):
    parser = get_default_parser()
    args = parser.parse_args()
    config = construct_config(args.opts)
    policy = SequentialExperts(
        config.WEIGHTS.NAV,
        config.WEIGHTS.GAZE,
        config.WEIGHTS.PLACE,
        device=config.DEVICE,
    )

    env = SpotMobileManipulationSeqEnv(
        config,
        spot,
        mask_rcnn_weights=config.WEIGHTS.MRCNN,
        mask_rcnn_device=config.DEVICE,
    )
    env.power_robot()
    time.sleep(1)
    count = Counter()
    try:
        for trip_idx in range(NUM_OBJECTS + 1):
            if trip_idx < NUM_OBJECTS:
                clutter_blacklist = [i for i in WAYPOINTS["clutter"] if count[i] >= 2]
                waypoint_name, waypoint = closest_clutter(
                    env.x, env.y, clutter_blacklist=clutter_blacklist
                )
                count[waypoint_name] += 1
                env.say("Going to " + waypoint_name + "to search for objects")
            else:
                env.say("Finished object rearrangement. Heading to dock.")
                waypoint = nav_target_from_waypoints("dock")
            observations = env.reset(waypoint=waypoint)
            policy.reset()
            done = False
            expert = Tasks.NAV
            while not done:
                base_action, arm_action = policy.act(observations, expert=expert)
                observations, _, done, info = env.step(
                    base_action=base_action, arm_action=arm_action
                )
                expert = info["correct_skill"]
                if trip_idx >= NUM_OBJECTS and expert != Tasks.NAV:
                    break

                # Print info
                stats = [f"{k}: {v}" for k, v in info.items()]
                print("\t".join(stats))

                # We reuse nav, so we have to reset it before we use it again.
                if expert != Tasks.NAV:
                    policy.nav_policy.reset()

        env.say("Executing automatic docking")
        dock_start_time = time.time()
        while time.time() - dock_start_time < 2:
            try:
                spot.dock(dock_id=520, home_robot=True)
            except:
                print("Dock not found... trying again")
                time.sleep(0.1)
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
    def __init__(self, config, spot: Spot, **kwargs):
        SpotBaseEnv.__init__(self, config, spot, **kwargs)

        # Nav
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)
        self.gaze_nav_target = None
        self.place_nav_target = None

        # Gaze
        self.locked_on_object_count = 0
        self.target_obj_id = config.TARGET_OBJ_ID

        # Place
        self.place_target = None
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.place_target_is_local = False

        # General
        self.max_episode_steps = 1000

    def reset(self, waypoint=None, *args, **kwargs):
        # Move arm to initial configuration (w/ gripper open)
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=0.75
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)
        self.spot.open_gripper()

        # Nav
        if waypoint is None:
            self.goal_xy = None
            self.goal_heading = None
        else:
            self.goal_xy, self.goal_heading = (waypoint[:2], waypoint[2])

        # Place
        self.place_target = mn.Vector3(-1.0, -1.0, -1.0)

        return SpotBaseEnv.reset(self)

    def step(self, *args, **kwargs):
        return SpotBaseEnv.step(self, *args, **kwargs)

    def get_observations(self):
        observations = self.get_nav_observation(self.goal_xy, self.goal_heading)
        observations.update(super().get_observations())
        observations["obj_start_sensor"] = self.get_place_sensor()

        return observations


class SpotMobileManipulationSeqEnv(
    SpotMobileManipulationBaseEnv, SpotGazeEnv, SpotPlaceEnv
):
    def __init__(self, config, spot, **kwargs):
        super().__init__(config, spot, **kwargs)
        self.current_task = Tasks.NAV
        self.nav_succ_count = 0

    def reset(self, *args, **kwargs):
        observations = super().reset(*args, **kwargs)
        self.current_task = Tasks.NAV
        self.target_obj_id = 0

        return observations

    def step(self, *args, **kwargs):
        if self.current_task == Tasks.PLACE:
            _, xy_dist, z_dist = self.get_place_distance()
            place = (
                xy_dist < self.config.SUCC_XY_DIST and z_dist < self.config.SUCC_Z_DIST
            )
        else:
            place = False

        grasp = (
            self.current_task == Tasks.GAZE
            and self.locked_on_object_count == self.config.OBJECT_LOCK_ON_NEEDED
        )

        if self.current_task == Tasks.PLACE:
            max_joint_movement_key = "MAX_JOINT_MOVEMENT_2"
        else:
            max_joint_movement_key = "MAX_JOINT_MOVEMENT"

        observations, reward, done, info = super().step(
            grasp=grasp,
            place=place,
            max_joint_movement_key=max_joint_movement_key,
            *args,
            **kwargs,
        )

        if self.current_task == Tasks.NAV and self.get_nav_success(
            observations, self.succ_distance, self.succ_angle
        ):
            # Make the robot stop moving
            self.spot.set_base_velocity(0.0, 0.0, 0.0, 1 / self.ctrl_hz)
            if not self.grasp_attempted:
                # Move arm to initial configuration
                cmd_id = self.spot.set_arm_joint_positions(
                    positions=self.initial_arm_joint_angles, travel_time=1
                )
                self.spot.block_until_arm_arrives(cmd_id, timeout_sec=1)
                self.current_task = Tasks.GAZE
                self.target_obj_id = None
            else:
                self.current_task = Tasks.PLACE
                self.say("Starting place")

        if self.current_task == Tasks.GAZE and self.grasp_attempted:
            self.current_task = Tasks.NAV
            # Determine where to go based on what object we're holding
            waypoint_name, waypoint = object_id_to_nav_waypoint(self.target_obj_id)
            self.say("Navigating to " + waypoint_name)
            self.place_target = place_target_from_waypoints(waypoint_name)
            self.goal_xy, self.goal_heading = (waypoint[:2], waypoint[2])

        info["correct_skill"] = self.current_task
        print("correct_skill", self.current_task)

        return observations, reward, done, info

    def get_success(self, observations):
        return self.place_attempted


if __name__ == "__main__":
    spot = Spot("RealSeqEnv")
    with spot.get_lease(hijack=True):
        main(spot)
