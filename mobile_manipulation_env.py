import magnum as mn
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

from base_env import SpotBaseEnv
from gaze_env import SpotGazeEnv
from place_env import SpotPlaceEnv
from real_policy import GazePolicy, NavPolicy, PlacePolicy
from utils import (
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
    place_target_from_waypoints,
)

PLACE_TARGET = place_target_from_waypoints("pillar_bin")
GAZE_DESTINATION = nav_target_from_waypoints("suitcase")
PLACE_DESTINATION = nav_target_from_waypoints("pillar")


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
    policy.reset()

    env = SpotMobileManipulationSeqEnv(
        config,
        spot,
        mask_rcnn_weights=config.WEIGHTS.MRCNN,
        mask_rcnn_device=config.DEVICE,
    )
    env.power_robot()
    observations = env.reset()
    done = False
    say("Starting episode")
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
        self.succ_angle = config.SUCCESS_ANGLE_DIST

        # Gaze
        self.locked_on_object_count = 0
        self.target_obj_id = config.TARGET_OBJ_ID

        # Place
        self.place_target = PLACE_TARGET
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.placed = False
        self.place_target_is_local = False

        # General
        self.gaze_nav_target = GAZE_DESTINATION
        self.place_nav_target = PLACE_DESTINATION
        self.max_episode_steps = 1000

    def reset(self):
        # Move arm to initial configuration (w/ gripper open)
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=0.75
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)
        self.spot.open_gripper()

        # Nav
        self.goal_xy, self.goal_heading = (
            self.gaze_nav_target[:2],
            self.gaze_nav_target[-1],
        )
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
        observations["obj_start_sensor"] = self.get_place_sensor()

        return observations


class SpotMobileManipulationSeqEnv(
    SpotMobileManipulationBaseEnv, SpotGazeEnv, SpotPlaceEnv
):
    def __init__(self, config, spot, **kwargs):
        super().__init__(config, spot, **kwargs)
        self.current_task = Tasks.NAV
        self.nav_succ_count = 0

    def reset(self):
        observations = super().reset()
        self.current_task = Tasks.NAV
        self.nav_succ_count = 0
        return observations

    def step(self, grasp=False, place=False, *args, **kwargs):
        _, xy_dist, z_dist = self.get_place_distance()
        place = (
            self.current_task == Tasks.PLACE
            and xy_dist < self.config.SUCC_XY_DIST
            and z_dist < self.config.SUCC_Z_DIST
        )

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
            self.nav_succ_count += 1
            if self.nav_succ_count == 1:
                self.spot.set_base_velocity(0.0, 0.0, 0.0, 1 / self.ctrl_hz)
                if not self.grasp_attempted:
                    self.current_task = Tasks.GAZE
                    self.goal_xy, self.goal_heading = (
                        self.place_nav_target[:2],
                        self.place_nav_target[2],
                    )
                else:
                    self.current_task = Tasks.PLACE
        else:
            self.nav_succ_count = 0

        if self.current_task == Tasks.GAZE and self.grasp_attempted:
            self.current_task = Tasks.NAV

        info["correct_skill"] = self.current_task
        print("correct_skill", self.current_task)

        return observations, reward, done, info

    def get_success(self, observations):
        return self.placed


if __name__ == "__main__":
    spot = Spot("RealSeqEnv")
    with spot.get_lease(hijack=True):
        main(spot)
