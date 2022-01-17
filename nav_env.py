import cv2
import gym
import numpy as np
import time

from bd_spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    image_response_to_cv2,
    scale_depth_img,
    wrap_heading,
)
from bd_spot_wrapper.utils import say
from nav_policy import NavPolicy

CTRL_HZ = 1.0
MAX_EPISODE_STEPS = 120
MAX_LIN_VEL = 0.5  # m/s
MAX_ANG_VEL = 0.523599  # 30 degrees/s, in radians
SUCCESS_DISTANCE = 0.3
SUCCESS_ANGLE_DIST = 0.0872665  # 5 radians
VEL_TIME = 0.5
NAV_WEIGHTS = "weights/two_cams_with_noise_seed4_ckpt.4.pth"
GOAL_XY = [-11/2, -11/2]
GOAL_HEADING = np.deg2rad(90)  # positive direction is CCW
MAX_DEPTH = 3.5
# POINTGOAL_UUID = "target_point_goal_gps_and_compass_sensor"
POINTGOAL_UUID = "pointgoal_with_gps_compass"


def main(spot):
    env = SpotNavEnv(spot)
    policy = NavPolicy(NAV_WEIGHTS, device="cpu")
    policy.reset()
    observations = env.reset(GOAL_XY, GOAL_HEADING)
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
        say("Environment is done.")
        time.sleep(20)
    finally:
        spot.power_off()


class SpotNavEnv(gym.Env):
    def __init__(self, spot: Spot):
        # Arrange Spot into initial configuration
        assert spot.spot_lease is not None, "Need motor control of Spot!"
        spot.power_on()
        say("Standing up")
        spot.blocking_stand()

        self.spot = spot
        self.num_steps = 0
        self.reset_ran = False
        self.goal_xy = None  # this is where the pointgoal is stored
        self.goal_heading = None  # this is where the goal heading is stored
        self.yaw = None

    def reset(self, goal_xy, goal_heading):
        # Reset parameters
        self.num_steps = 0
        self.reset_ran = True
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        assert len(self.goal_xy) == 2

        observations = self.get_observations()

        return observations

    def step(self, action):
        """
        Moves the arm and returns updated observations
        :param action: np.array of radians denoting how much each joint is to be moved
        :return:
        """
        assert self.reset_ran, ".reset() must be called first!"

        # Command velocities using the input action
        x_vel, ang_vel = action
        x_vel = np.clip(x_vel, -1, 1) * MAX_LIN_VEL
        ang_vel = np.clip(ang_vel, -1, 1) * MAX_ANG_VEL

        # TODO: HACK!! Need to do this smarter
        self.spot.set_base_velocity(x_vel, 0.0, ang_vel, VEL_TIME)  # 0.0 for y_vel
        time.sleep(VEL_TIME)

        observations = self.get_observations()
        self.print_stats(observations)

        done = self.num_steps == MAX_EPISODE_STEPS or self.get_success(observations)

        # Don't need reward or info
        reward, info = None, {}
        return observations, reward, done, info

    @staticmethod
    def get_success(obs):
        # Is the agent at the goal?
        dist_to_goal, _ = obs[POINTGOAL_UUID]
        at_goal = dist_to_goal < SUCCESS_DISTANCE
        good_heading = abs(obs["goal_heading"][0]) < SUCCESS_ANGLE_DIST
        print(f"at_goal: {at_goal} good_heading: {good_heading}")
        return at_goal and good_heading

    def print_stats(self, observations):
        rho, theta = observations[POINTGOAL_UUID]
        print(
            f"Dist to goal: {rho:.2f}\t"
            f"theta: {np.rad2deg(theta):.2f}\t"
            f"x: {self.x:.2f}\t"
            f"y: {self.y:.2f}\t"
            f"yaw: {np.rad2deg(self.yaw):.2f}\t"
        )

    @staticmethod
    def transform_depth_response(depth_response):
        depth_cv2 = image_response_to_cv2(depth_response)
        rotated_depth_cv2 = np.rot90(depth_cv2, k=3)
        scaled_depth_cv2 = scale_depth_img(rotated_depth_cv2, max_depth=MAX_DEPTH)
        resized_depth_cv2 = cv2.resize(scaled_depth_cv2, (120, 212))
        unsqueezed_depth_cv2 = resized_depth_cv2.reshape(
            [*resized_depth_cv2.shape[:2], 1]
        )
        return unsqueezed_depth_cv2

    def get_observations(self):
        sources = [SpotCamIds.FRONTLEFT_DEPTH, SpotCamIds.FRONTRIGHT_DEPTH]
        image_responses = self.spot.get_image_responses(sources)
        spot_left_depth, spot_right_depth = [
            self.transform_depth_response(r) for r in image_responses
        ]

        x, y, self.yaw = self.spot.get_xy_yaw()
        curr_xy = np.array([x, y], dtype=np.float32)
        rho = np.linalg.norm(curr_xy - self.goal_xy)
        theta = np.arctan2(self.goal_xy[1] - y, self.goal_xy[0] - x) - self.yaw
        rho_theta = np.array([rho, wrap_heading(theta)], dtype=np.float32)

        goal_heading = -np.array([self.goal_heading - self.yaw], dtype=np.float32)
        observations = {
            "spot_left_depth": spot_left_depth,
            "spot_right_depth": spot_right_depth,
            POINTGOAL_UUID: rho_theta,
            "goal_heading": goal_heading,
        }
        self.x, self.y = x, y

        return observations

if __name__ == "__main__":
    spot = Spot("RealNavEnv")
    with spot.get_lease():
        main(spot)
