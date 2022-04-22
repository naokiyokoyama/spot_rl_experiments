import os
import os.path as osp
import sys

this_dir = osp.dirname(osp.abspath(__file__))
base_dir = osp.join(this_dir, "spot_rl")
bin_dir = osp.join(os.environ["CONDA_PREFIX"], "bin")

orig_to_alias = {
    "envs.gaze_env": "spot_rl_gaze_env",
    "envs.mobile_manipulation_env": "spot_rl_mobile_manipulation_env",
    "envs.nav_env": "spot_rl_nav_env",
    "envs.place_env": "spot_rl_place_env",
    "baselines.go_to_waypoint": "spot_rl_go_to_waypoint",
    "utils.autodock": "spot_rl_autodock",
    "utils.waypoint_recorder": "spot_rl_waypoint_recorder",
    "ros_img_vis": "spot_rl_ros_img_vis",
}

print("Generating executables...")
for orig, alias in orig_to_alias.items():
    exe_path = osp.join(bin_dir, alias)
    data = f"#!/usr/bin/env bash \n{sys.executable} -m spot_rl.{orig} $@\n"
    with open(exe_path, "w") as f:
        f.write(data)
    os.chmod(exe_path, 33277)
    print("Added:", alias)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("THESE EXECUTABLES ARE ONLY VISIBLE TO THE CURRENT CONDA ENV!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
