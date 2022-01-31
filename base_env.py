import time

import cv2
import gym
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    image_response_to_cv2,
    scale_depth_img,
    wrap_heading,
)
from spot_wrapper.utils import say

SPOT_IMAGE_SOURCES = [
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.HAND_DEPTH,
    SpotCamIds.HAND_COLOR,
]

FRONT_DEPTH_TOPIC = "/spot_cams/filtered_front_depth"
HAND_DEPTH_TOPIC = f"/spot_cams/{SpotCamIds.HAND_DEPTH}"
HAND_COLOR_TOPIC = f"/spot_cams/{SpotCamIds.HAND_COLOR}"

CTRL_HZ = 2
MAX_EPISODE_STEPS = 100


# Base action params
MAX_LIN_VEL = 0.5  # m/s
MAX_ANG_VEL = 0.523599  # 30 degrees/s, in radians
VEL_TIME = 0.5

# Arm action params
MAX_JOINT_MOVEMENT = 0.0698132
INITIAL_ARM_JOINT_ANGLES = np.deg2rad([0, -170, 120, 0, 75, 0])
ARM_LOWER_LIMITS = np.deg2rad([-45, -180, 0, 0, -90, 0])
ARM_UPPER_LIMITS = np.deg2rad([45, -45, 135, 0, 90, 0])
JOINT_BLACKLIST = ["arm0.el0", "arm0.wr1"]  # joints we can't control
ACTUALLY_MOVE_ARM = True


def pad_action(action):
    """We only control 4 out of 6 joints; add zeros to non-controllable indices."""
    return np.array([*action[:3], 0.0, action[3], 0.0])


class SpotBaseEnv(gym.Env):
    def __init__(self, spot: Spot):
        self.spot = spot
        self.num_steps = 0
        self.reset_ran = False

        self.cv_bridge = CvBridge()

        # ROS subscribers
        rospy.init_node("spot_reality_gym", disable_signals=True)  # enable Ctrl + C
        rospy.Subscriber(FRONT_DEPTH_TOPIC, Image, self.front_depth_cb)
        rospy.Subscriber(HAND_DEPTH_TOPIC, Image, self.hand_depth_cb)
        rospy.Subscriber(HAND_COLOR_TOPIC, Image, self.hand_color_cb)
        self.front_depth_img = None
        self.hand_depth_img = None
        self.hand_color_img = None

        # General environment parameters
        self.ctrl_hz = CTRL_HZ
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.last_execution = time.time()

        # Robot state parameters
        self.x, self.y, self.yaw = None, None, None
        self.current_arm_pose = None

        # Base action parameters
        self.max_lin_vel = MAX_LIN_VEL
        self.max_ang_vel = MAX_ANG_VEL
        self.vel_time = VEL_TIME

        # Arm action parameters
        self.initial_arm_joint_angles = INITIAL_ARM_JOINT_ANGLES
        self.max_joint_movement = MAX_JOINT_MOVEMENT
        self.max_ang_vel = MAX_ANG_VEL
        self.vel_time = VEL_TIME
        self.actually_move_arm = ACTUALLY_MOVE_ARM
        self.arm_lower_limits = ARM_LOWER_LIMITS
        self.arm_upper_limits = ARM_UPPER_LIMITS

        # Arrange Spot into initial configuration
        assert spot.spot_lease is not None, "Need motor control of Spot!"
        spot.power_on()
        say("Standing up")
        spot.blocking_stand()

    def front_depth_cb(self, msg):
        self.front_depth_img = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")

    def hand_depth_cb(self, msg):
        self.hand_depth_img = self.cv_bridge.imgmsg_to_cv2(msg, "mono16")

    def hand_color_cb(self, msg):
        self.hand_color_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def reset(self, *args, **kwargs):
        # Reset parameters
        self.num_steps = 0
        self.reset_ran = True

        # Save latest position of each arm joint
        self.get_arm_joints()

        observations = self.get_observations()
        return observations

    def step(self, base_action=None, arm_action=None, grasp=False, place=False):
        """Moves the arm and returns updated observations

        :param base_action: np.array of velocities (lineaer, angular)
        :param arm_action: np.array of radians denoting how each joint is to be moved
        :param grasp: whether to call the grasp_center_of_hand_depth() METHOD
        :param place: whether to call the open_gripper() method
        :return:
        """
        assert self.reset_ran, ".reset() must be called first!"
        assert base_action is not None or arm_action is not None, "Must provide action."

        # Pause until enough time has passed
        while time.time() < self.last_execution + 1 / self.ctrl_hz:
            pass
        print("Env Hz:", 1 / (time.time() - self.last_execution))
        self.last_execution = time.time()

        if base_action is not None:
            # Command velocities using the input action
            x_vel, ang_vel = base_action
            x_vel = np.clip(x_vel, -1, 1) * self.max_lin_vel
            ang_vel = np.clip(ang_vel, -1, 1) * self.max_ang_vel
            # No horizontal velocity
            self.spot.set_base_velocity(x_vel, 0.0, ang_vel, self.vel_time)

        if arm_action is not None:
            # Insert zeros for joints we don't control
            padded_action = pad_action(arm_action)
            padded_action = np.clip(padded_action, -1.0, 1.0) * self.max_joint_movement
            target_positions = np.clip(
                self.current_arm_pose + padded_action,
                self.arm_lower_limits,
                self.arm_upper_limits,
            )
            if self.actually_move_arm:
                _ = self.spot.set_arm_joint_positions(
                    positions=target_positions, travel_time=1 / self.ctrl_hz
                )

        if grasp:
            # The following cmd is blocking
            self.spot.grasp_center_of_hand_depth()
            # Return to pre-grasp joint positions after grasp
            cmd_id = self.spot.set_arm_joint_positions(
                positions=self.initial_arm_joint_angles, travel_time=1.0
            )
            self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)

        if place:
            self.spot.open_gripper()

        observations = self.get_observations()

        timeout = self.num_steps == self.max_episode_steps
        done = timeout or self.get_success(observations)

        # Don't need reward or info
        reward, info = None, {}
        return observations, reward, done, info

    @staticmethod
    def get_nav_success(observations, success_distance, success_angle):
        # Is the agent at the goal?
        dist_to_goal, _ = observations["target_point_goal_gps_and_compass_sensor"]
        at_goal = dist_to_goal < success_distance
        good_heading = abs(observations["goal_heading"][0]) < success_angle
        return at_goal and good_heading

    def print_nav_stats(self, observations):
        rho, theta = observations["target_point_goal_gps_and_compass_sensor"]
        print(
            f"Dist to goal: {rho:.2f}\t"
            f"theta: {np.rad2deg(theta):.2f}\t"
            f"x: {self.x:.2f}\t"
            f"y: {self.y:.2f}\t"
            f"yaw: {np.rad2deg(self.yaw):.2f}\t"
            f"gh: {np.rad2deg(observations['goal_heading'][0]):.2f}\t"
        )

    def get_nav_observation(self, goal_xy, goal_heading):
        observations = {}
        # Get visual observations
        front_depth = cv2.resize(
            self.front_depth_img, (120 * 2, 212), interpolation=cv2.INTER_AREA
        )
        front_depth = np.float32(front_depth) / 255.0
        # Add dimension for channel (unsqueeze)
        cv2.imwrite("test.png", np.uint8(front_depth * 255))
        front_depth = front_depth.reshape(*front_depth.shape[:2], 1)
        observations["spot_left_depth"], observations["spot_right_depth"] = np.split(
            front_depth, 2, 1
        )

        # Get rho theta observation
        self.x, self.y, self.yaw = self.spot.get_xy_yaw()
        curr_xy = np.array([self.x, self.y], dtype=np.float32)
        rho = np.linalg.norm(curr_xy - goal_xy)
        theta = np.arctan2(goal_xy[1] - self.y, goal_xy[0] - self.x) - self.yaw
        rho_theta = np.array([rho, wrap_heading(theta)], dtype=np.float32)

        # Get goal heading observation
        goal_heading_ = -np.array([goal_heading - self.yaw], dtype=np.float32)
        observations["target_point_goal_gps_and_compass_sensor"] = rho_theta
        observations["goal_heading"] = goal_heading_

        return observations

    def get_arm_joints(self):
        # Get proprioception inputs
        arm_proprioception = self.spot.get_arm_proprioception().values()
        self.current_arm_pose = np.array([j.position.value for j in arm_proprioception])
        joints = np.array(
            [
                j.position.value
                for j in arm_proprioception
                if j.name not in JOINT_BLACKLIST
            ],
            dtype=np.float32,
        )

        return joints

    def get_observations(self):
        raise NotImplementedError

    def get_success(self, observations):
        raise NotImplementedError
