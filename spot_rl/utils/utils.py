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


def closest_clutter(x, y, clutter_blacklist=None):
    if clutter_blacklist is None:
        clutter_blacklist = []
    clutter_locations = [
        (np.array(nav_target_from_waypoints(w)[:2]), w)
        for w in WAYPOINTS["clutter"]
        if w not in clutter_blacklist
    ]
    xy = np.array([x, y])
    dist_to_clutter = lambda i: np.linalg.norm(i[0] - xy)
    _, waypoint_name = sorted(clutter_locations, key=dist_to_clutter)[0]
    return waypoint_name, nav_target_from_waypoints(waypoint_name)


def object_id_to_nav_waypoint(object_id):
    if isinstance(object_id, str):
        for k, v in WAYPOINTS["object_targets"].items():
            if v[0] == object_id:
                object_id = int(k)
                break
        if isinstance(object_id, str):
            KeyError(f"{object_id} not a valid class name!")
    place_nav_target_name = WAYPOINTS["object_targets"][object_id][1]
    return place_nav_target_name, nav_target_from_waypoints(place_nav_target_name)


def object_id_to_object_name(object_id):
    return WAYPOINTS["object_targets"][object_id][0]
