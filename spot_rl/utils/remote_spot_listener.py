"""
The code here should be by the Core only. This will relay any received commands straight
to the robot from the Core via Ethernet.
"""

import argparse
import os.path as osp
import subprocess
import time

import numpy as np
import rospy
from spot_wrapper.spot import Spot
from std_msgs.msg import Bool, String

ROBOT_CMD_TOPIC = "/remote_robot_cmd"
CMD_ENDED_TOPIC = "/remote_robot_cmd_ended"
INIT_REMOTE_ROBOT = "/init_remote_robot"
KILL_REMOTE_ROBOT = "/kill_remote_robot"


class RemoteSpotListener:
    def __init__(self, spot):
        self.spot = spot
        assert spot.spot_lease is not None, "Need motor control of Spot!"

        # This subscriber executes received cmds
        rospy.Subscriber(ROBOT_CMD_TOPIC, String, self.execute_cmd, queue_size=1)

        # This publisher signals if a cmd has finished
        self.pub = rospy.Publisher(CMD_ENDED_TOPIC, Bool, queue_size=1)

        # This subscriber will kill the listener
        rospy.Subscriber(KILL_REMOTE_ROBOT, Bool, self.kill_remote_robot, queue_size=1)

    def execute_cmd(self, msg):
        values = msg.data.split(";")
        method_name, args = values[0], values[1:]
        method = eval("self.spot." + method_name)
        method(*[eval(i) for i in args])
        self.pub.publish(True)

    def kill_remote_robot(self, msg):
        raise RuntimeError


class RemoteSpotMaster:
    def __init__(self):
        # This subscriber executes received cmds
        rospy.Subscriber(INIT_REMOTE_ROBOT, Bool, self.init_remote_robot, queue_size=1)
        self.remote_robot_killer = rospy.Publisher(
            KILL_REMOTE_ROBOT, Bool, queue_size=1
        )

    def init_remote_robot(self, msg):
        self.remote_robot_killer.publish(True)
        time.sleep(1)
        # Kill any surviving listeners, if they exist...
        try:
            subprocess.check_call(
                "tmux kill-session -t remote_robot_listener", shell=True
            )
        except:
            pass

        # Run this script again in a separate tmux, but not as master
        python = "/home/spot/anaconda3/envs/spot_ros/bin/python"
        if not osp.isfile(python):
            python = python.replace("anaconda3", "miniconda3")
            assert osp.isfile(python), "Can't find python interpreter!"
        this_file = osp.abspath(__file__)
        subprocess.check_call(
            f"tmux new -s remote_robot_listener -d '{python} {this_file}'", shell=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", action="store_true")
    is_master = parser.parse_args().master

    if is_master:
        RemoteSpotMaster()
        rospy.spin()
    else:
        spot = Spot("RemoteSpotListener")
        with spot.get_lease(hijack=True):
            try:
                rsl = RemoteSpotListener(spot)
                rospy.spin()
            finally:
                spot.power_off()
