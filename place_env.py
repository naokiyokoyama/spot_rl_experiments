import time

import magnum as mn
import numpy as np
from spot_wrapper.spot import Spot
from spot_wrapper.utils import say

from base_env import SpotBaseEnv
from real_policy import PlacePolicy
from utils import construct_config, get_default_parser, place_target_from_waypoints


def main(spot):
    parser = get_default_parser()
    parser.add_argument("-p", "--place_target")
    parser.add_argument("-w", "--waypoint")
    parser.add_argument("-l", "--target_is_local", action="store_true")
    args = parser.parse_args()
    config = construct_config(args.opts)

    if args.waypoint is not None:
        assert not args.target_is_local
        place_target = place_target_from_waypoints(args.waypoint)
    else:
        assert args.place_target is not None
        place_target = [float(i) for i in args.place_target.split(",")]
    env = SpotPlaceEnv(config, spot, place_target, args.target_is_local)
    env.power_robot()
    policy = PlacePolicy(config.WEIGHTS.PLACE, device=config.DEVICE)
    policy.reset()
    observations = env.reset()
    done = False
    say("Starting episode")
    try:
        while not done:
            action = policy.act(observations)
            observations, _, done, _ = env.step(arm_action=action)
    finally:
        spot.power_off()


class SpotPlaceEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot, place_target, target_is_local=False):
        super().__init__(config, spot)
        self.place_target = np.array(place_target)
        self.place_target_is_local = target_is_local
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
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
        _, xy_dist, z_dist = self.get_place_distance()
        place = xy_dist < self.config.SUCC_XY_DIST and z_dist < self.config.SUCC_Z_DIST
        return super().step(place=place, *args, **kwargs)

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "obj_start_sensor": self.get_place_sensor(),
        }

        return observations

    def get_place_sensor(self):
        # The place goal should be provided relative to the local robot frame given that
        # the robot is at the place receptacle
        gripper_T_base = self.get_in_gripper_tf()
        base_T_gripper = gripper_T_base.inverted()
        base_frame_place_target = self.get_base_frame_place_target()
        hab_place_target = self.spot2habitat_translation(base_frame_place_target)
        gripper_pos = base_T_gripper.transform_point(hab_place_target)

        return gripper_pos

    def get_base_frame_place_target(self):
        if self.place_target_is_local:
            base_frame_place_target = self.place_target
        else:
            base_frame_place_target = self.get_target_in_base_frame(self.place_target)
        return base_frame_place_target

    def get_place_distance(self):
        gripper_T_base = self.get_in_gripper_tf()
        base_frame_gripper_pos = np.array(gripper_T_base.translation)
        base_frame_place_target = self.get_base_frame_place_target()
        hab_place_target = self.spot2habitat_translation(base_frame_place_target)
        hab_place_target = np.array(hab_place_target)
        place_dist = np.linalg.norm(hab_place_target - base_frame_gripper_pos)
        xy_dist = np.linalg.norm(
            hab_place_target[[0, 2]] - base_frame_gripper_pos[[0, 2]]
        )
        z_dist = abs(hab_place_target[1] - base_frame_gripper_pos[1])
        return place_dist, xy_dist, z_dist

    def get_in_gripper_tf(self):
        position, rotation = self.spot.get_base_transform_to("link_wr1")
        wrist_T_base = self.spot2habitat_transform(position, rotation)
        gripper_T_base = wrist_T_base @ mn.Matrix4.translation(self.ee_gripper_offset)

        return gripper_T_base

    def get_target_in_base_frame(self, place_target):
        global_T_local = self.curr_transform.inverted()
        local_place_target = np.array(global_T_local.transform_point(place_target))
        local_place_target[1] *= -1  # Still not sure why this is necessary

        return local_place_target


if __name__ == "__main__":
    spot = Spot("RealPlaceEnv")
    with spot.get_lease(hijack=True):
        main(spot)
