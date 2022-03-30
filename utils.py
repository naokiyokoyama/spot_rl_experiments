import argparse

import numpy as np
import yaml
from yacs.config import CfgNode as CN

DEFAULT_CONFIG = "configs/config.yaml"
WAYPOINTS_YAML = "waypoints.yaml"


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opts", nargs="*", default=[])
    return parser


def construct_config(opts):
    config = CN()
    config.set_new_allowed(True)
    config.merge_from_file(DEFAULT_CONFIG)
    config.merge_from_list(opts)

    return config


def nav_target_from_waypoints(waypoint):
    with open(WAYPOINTS_YAML) as f:
        waypoints = yaml.safe_load(f)
    goal_x, goal_y, goal_heading = waypoints[waypoint]
    return goal_x, goal_y, np.deg2rad(goal_heading)
