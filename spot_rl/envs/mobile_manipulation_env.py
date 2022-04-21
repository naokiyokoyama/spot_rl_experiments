import os
import time
from collections import Counter

import magnum as mn
import numpy as np
from gaze_env import SpotGazeEnv
from spot_wrapper.spot import Spot

from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import GazePolicy, MixerPolicy, NavPolicy, PlacePolicy
from spot_rl.utils.remote_spot import RemoteSpot
from spot_rl.utils.utils import (
    WAYPOINTS,
    closest_clutter,
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
    object_id_to_nav_waypoint,
    place_target_from_waypoints,
)

CLUTTER_AMOUNTS = {
    "whiteboard": 3,
    "black_box": 2,
    "white_box": 3,
}
NUM_OBJECTS = np.sum(list(CLUTTER_AMOUNTS.values()))
DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


def main(spot):
    parser = get_default_parser()
    parser.add_argument("-m", "--use-mixer", action="store_true")
    args = parser.parse_args()
    config = construct_config(args.opts)
    if args.use_mixer:
        policy = MixerPolicy(
            config.WEIGHTS.MIXER,
            config.WEIGHTS.NAV,
            config.WEIGHTS.GAZE,
            config.WEIGHTS.PLACE,
            device=config.DEVICE,
        )
        env_class = SpotMobileManipulationBaseEnv
    else:
        policy = SequentialExperts(
            config.WEIGHTS.NAV,
            config.WEIGHTS.GAZE,
            config.WEIGHTS.PLACE,
            device=config.DEVICE,
        )
        env_class = SpotMobileManipulationSeqEnv

    env = env_class(
        config,
        spot,
        mask_rcnn_weights=config.WEIGHTS.MRCNN,
        mask_rcnn_device=config.DEVICE,
    )
    env.power_robot()
    time.sleep(1)
    count = Counter()
    for trip_idx in range(NUM_OBJECTS + 1):
        if trip_idx < NUM_OBJECTS:
            # 2 objects per receptacle
            clutter_blacklist = [
                i for i in WAYPOINTS["clutter"] if count[i] >= CLUTTER_AMOUNTS[i]
            ]
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
        if args.use_mixer:
            expert = None
        else:
            expert = Tasks.NAV
        while not done:
            if expert is None:
                base_action, arm_action = policy.act(observations)
            else:
                base_action, arm_action = policy.act(observations, expert=expert)
            observations, _, done, info = env.step(
                base_action=base_action, arm_action=arm_action
            )
            if expert is not None:
                expert = info["correct_skill"]

            if trip_idx >= NUM_OBJECTS and env.get_nav_success(
                observations, 0.3, np.deg2rad(10)
            ):
                # The robot has arrived back at the dock
                break

            # Print info
            # stats = [f"{k}: {v}" for k, v in info.items()]
            # print("\t".join(stats))

            # We reuse nav, so we have to reset it before we use it again.
            if expert is not None and expert != Tasks.NAV:
                policy.nav_policy.reset()

    env.say("Executing automatic docking")
    dock_start_time = time.time()
    while time.time() - dock_start_time < 2:
        try:
            spot.dock(dock_id=DOCK_ID, home_robot=True)
        except:
            print("Dock not found... trying again")
            time.sleep(0.1)


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


class SpotMobileManipulationBaseEnv(SpotGazeEnv):
    def __init__(self, config, spot: Spot, **kwargs):
        super().__init__(config, spot, **kwargs)

        # Nav
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)
        self.gaze_nav_target = None
        self.place_nav_target = None

        # Gaze
        self.locked_on_object_count = 0
        self.target_obj_name = config.TARGET_OBJ_NAME

        # Place
        self.place_target = None
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.place_target_is_local = False

        # General
        self.max_episode_steps = 1000
        self.navigating_to_place = False

    def reset(self, waypoint=None, *args, **kwargs):
        # Move arm to initial configuration (w/ gripper open)
        self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=0.75
        )
        # Wait for arm to arrive to position
        time.sleep(0.75)
        self.spot.open_gripper()

        # Nav
        if waypoint is None:
            self.goal_xy = None
            self.goal_heading = None
        else:
            self.goal_xy, self.goal_heading = (waypoint[:2], waypoint[2])

        # Place
        self.place_target = mn.Vector3(-1.0, -1.0, -1.0)

        # General
        self.navigating_to_place = False

        return SpotBaseEnv.reset(self)

    def step(self, *args, **kwargs):
        _, xy_dist, z_dist = self.get_place_distance()
        place = xy_dist < self.config.SUCC_XY_DIST and z_dist < self.config.SUCC_Z_DIST
        grasp = (
            self.locked_on_object_count >= self.config.OBJECT_LOCK_ON_NEEDED
            and not self.grasp_attempted
            and 0.2 < self.target_object_distance < 1.0
        )
        if (
            not grasp
            and not self.grasp_attempted
            and self.locked_on_object_count >= self.config.OBJECT_LOCK_ON_NEEDED
        ):
            print(f"Can't grasp: object is {self.target_object_distance} m far")

        if self.grasp_attempted:
            max_joint_movement_key = "MAX_JOINT_MOVEMENT_2"
        else:
            max_joint_movement_key = "MAX_JOINT_MOVEMENT"

        observations, reward, done, info = SpotBaseEnv.step(
            self,
            grasp=grasp,
            place=place,
            max_joint_movement_key=max_joint_movement_key,
            *args,
            **kwargs,
        )

        if self.grasp_attempted and not self.navigating_to_place:
            # Determine where to go based on what object we've just grasped
            waypoint_name, waypoint = object_id_to_nav_waypoint(self.target_obj_name)
            self.say("Navigating to " + waypoint_name)
            self.place_target = place_target_from_waypoints(waypoint_name)
            self.goal_xy, self.goal_heading = (waypoint[:2], waypoint[2])
            self.navigating_to_place = True

        return observations, reward, done, info

    def get_observations(self):
        observations = self.get_nav_observation(self.goal_xy, self.goal_heading)
        rho = observations["target_point_goal_gps_and_compass_sensor"][0]
        goal_heading = observations["goal_heading"][0]
        self.use_mrcnn = rho < 2 and abs(goal_heading) < np.deg2rad(60)
        observations.update(super().get_observations())
        observations["obj_start_sensor"] = self.get_place_sensor()

        return observations

    def get_success(self, observations):
        return self.place_attempted


class SpotMobileManipulationSeqEnv(SpotMobileManipulationBaseEnv):
    def __init__(self, config, spot, **kwargs):
        super().__init__(config, spot, **kwargs)
        self.current_task = Tasks.NAV

    def reset(self, *args, **kwargs):
        observations = super().reset(*args, **kwargs)
        self.current_task = Tasks.NAV
        self.target_obj_name = 0

        return observations

    def step(self, *args, **kwargs):
        pre_step_navigating_to_place = self.navigating_to_place
        observations, reward, done, info = super().step(*args, **kwargs)

        if self.current_task == Tasks.NAV and self.get_nav_success(
            observations, self.succ_distance, self.succ_angle
        ):
            if not self.grasp_attempted:
                self.current_task = Tasks.GAZE
                self.target_obj_name = None
            else:
                self.current_task = Tasks.PLACE
                self.say("Starting place")

        if not pre_step_navigating_to_place and self.navigating_to_place:
            # This means that the Gaze task has just ended
            self.current_task = Tasks.NAV

        info["correct_skill"] = self.current_task

        self.use_mrcnn = self.current_task == Tasks.GAZE

        return observations, reward, done, info


if __name__ == "__main__":
    use_remote = True
    if use_remote:
        spot = RemoteSpot("RealSeqEnv")
    else:
        spot = Spot("RealSeqEnv")
    if use_remote:
        try:
            main(spot)
        finally:
            spot.power_off()
    else:
        with spot.get_lease(hijack=True):
            try:
                main(spot)
            finally:
                spot.power_off()
