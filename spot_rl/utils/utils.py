import argparse

import numpy as np
import yaml
from yacs.config import CfgNode as CN

DEFAULT_CONFIG = "configs/config.yaml"
WAYPOINTS_YAML = "configs/waypoints.yaml"
with open(WAYPOINTS_YAML) as f:
    WAYPOINTS = yaml.safe_load(f)


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
    goal_x, goal_y, goal_heading = WAYPOINTS[waypoint]
    return goal_x, goal_y, np.deg2rad(goal_heading)


def place_target_from_waypoints(waypoint):
    return np.array(WAYPOINTS["place_targets"][waypoint])


def obj_to_receptacle(object_name):
    return WAYPOINTS["object_targets"][object_name]


def closest_clutter(x, y):
    clutter_locations = [
        np.array(nav_target_from_waypoints(w)[:2]) for w in WAYPOINTS["clutter"]
    ]
    xy = np.array([x, y])
    dist_to_clutter = lambda i: np.linalg.norm(i - xy)
    return sorted(clutter_locations, key=dist_to_clutter)[0]
