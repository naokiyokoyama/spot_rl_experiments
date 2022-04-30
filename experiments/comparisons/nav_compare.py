import time

import numpy as np
from spot_wrapper.spot import Spot, wrap_heading

from spot_rl.envs.nav_env import SpotNavEnv
from spot_rl.real_policy import NavPolicy
from spot_rl.utils.utils import construct_config

goal_waypoint = 4.0, 0.0, np.pi / 2
start_waypoint = 1.0, 0.0, -np.pi / 2


def main(spot):
    config = construct_config([])
    policy = NavPolicy(config.WEIGHTS.NAV, device=config.DEVICE)

    env = SpotNavEnv(config, spot)
    env.power_robot()

    return_to_start(spot, start_waypoint)
    time.sleep(2)

    msg = []

    for idx, nav_func in enumerate([learned_navigate, baseline_navigate]):
        msg.append(f"Method {idx}:")
        for _ in range(3):
            st = time.time()
            nav_func(spot, goal_waypoint, policy=policy, env=env)
            msg.append(time.time() - st)
            spot.set_base_velocity(0, 0, 0, 1)
            time.sleep(1)
            return_to_start(spot, start_waypoint)

    print("\n".join([str(i) for i in msg]))


def baseline_navigate(spot, waypoint, **kwargs):
    goal_x, goal_y, goal_heading = waypoint
    cmd_id = spot.set_base_position(
        x_pos=goal_x,
        y_pos=goal_y,
        yaw=goal_heading,
        end_time=100,
        max_fwd_vel=0.5,
        max_hor_vel=0.05,
        max_ang_vel=np.deg2rad(30),
    )
    cmd_status = None
    success = False
    while not success:
        if cmd_status != 1:
            time.sleep(0.5)
            feedback_resp = spot.get_cmd_feedback(cmd_id)
            cmd_status = (
                feedback_resp.feedback.synchronized_feedback.mobility_command_feedback
            ).se2_trajectory_feedback.status
        else:
            cmd_id = spot.set_base_position(
                x_pos=goal_x,
                y_pos=goal_y,
                yaw=goal_heading,
                end_time=100,
                max_fwd_vel=0.5,
                max_hor_vel=0.1,
                max_ang_vel=np.deg2rad(30),
            )

        x, y, yaw = spot.get_xy_yaw()
        dist = np.linalg.norm(np.array([x, y]) - np.array([goal_x, goal_y]))
        heading_diff = abs(wrap_heading(goal_heading - yaw))
        success = dist < 0.3 and heading_diff < np.deg2rad(5)
    # print("succ", dist, heading_diff)


def learned_navigate(spot, waypoint, policy, env):
    goal_x, goal_y, goal_heading = waypoint
    observations = env.reset((goal_x, goal_y), goal_heading)
    done = False
    policy.reset()
    while not done:
        st = time.time()
        action = policy.act(observations)
        # print("policy act time:", time.time() - st)

        st = time.time()
        observations, _, done, _ = env.step(base_action=action)
        # print("env step time:", time.time() - st)


def return_to_start(spot, waypoint):
    goal_x, goal_y, goal_heading = waypoint
    spot.set_base_position(
        x_pos=goal_x, y_pos=goal_y, yaw=goal_heading, end_time=100, blocking=True
    )


if __name__ == "__main__":
    spot = Spot("NavCompare")
    with spot.get_lease():
        main(spot)
