# TODO: Support the following:
# self.x, self.y, self.yaw = self.spot.xy_yaw_global_to_home(
# position, rotation = self.spot.get_base_transform_to("link_wr1")


"""
This class allows you to control Spot as if you had a lease to actuate its motors,
but will actually just relay any motor commands to the robot's onboard Core. The Core
is the one that actually possesses the lease and sends motor commands to Spot via
Ethernet (faster, more reliable).

The message relaying is executed with ROS topic publishing / subscribing.

Very hacky.
"""

import time

import rospy
from spot_wrapper.spot import Spot
from std_msgs.msg import Bool, String

ROBOT_CMD_TOPIC = "/remote_robot_cmd"
CMD_ENDED_TOPIC = "/remote_robot_cmd_ended"
KILL_REMOTE_ROBOT = "/kill_remote_robot"


class RemoteSpot(Spot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This determines whether the Core has confirmed the last cmd has ended
        self.cmd_ended = False
        # This subscriber updates the above attribute
        rospy.Subscriber(CMD_ENDED_TOPIC, Bool, self.cmd_ended_callback, queue_size=1)

        # This publisher sends the desired command to the Core
        self.pub = rospy.Publisher(ROBOT_CMD_TOPIC, String, queue_size=1)
        self.remote_robot_killer = rospy.Publisher(
            KILL_REMOTE_ROBOT, Bool, queue_size=1
        )

    def cmd_ended_callback(self, msg):
        self.cmd_ended = msg.data

    def send_cmd(self, cmd_name, *args):
        cmd_with_args_str = ";".join([cmd_name] + [str(i) for i in args])
        self.pub.publish(cmd_with_args_str)

    @staticmethod
    def array2str(arr):
        return f"np.array([{','.join([str(i) for i in arr])}])"

    def blocking(self, timeout):
        start_time = time.time()
        self.cmd_ended = False
        while not self.cmd_ended and time.time() < start_time + timeout:
            # We need to block until we receive confirmation from the Core that the
            # grasp has ended
            time.sleep(0.1)
        self.cmd_ended = False

        if time.time() > start_time + timeout:
            return False

        return True

    def grasp_hand_depth(self, pixel_xy=None, timeout=10):
        if pixel_xy is None:
            self.send_cmd("grasp_hand_depth")
        else:
            self.send_cmd("grasp_hand_depth", self.array2str(pixel_xy))

        return self.blocking(timeout)

    def set_arm_joint_positions(
        self, positions, travel_time=1.0, max_vel=2.5, max_acc=15
    ):
        self.send_cmd(
            "set_arm_joint_positions",
            self.array2str(positions),
            travel_time,
            max_vel,
            max_acc,
        )

    def open_gripper(self):
        self.send_cmd("open_gripper")

    def set_base_velocity(
        self, x_vel, y_vel, ang_vel, vel_time, disable_obstacle_avoidance=False
    ):
        self.send_cmd(
            "set_base_velocity",
            x_vel,
            y_vel,
            ang_vel,
            vel_time,
            disable_obstacle_avoidance,
        )

    def power_on(self, *args, **kwargs):
        self.send_cmd("power_on")
        return self.blocking(20)

    def blocking_stand(self, *args, **kwargs):
        # Warning: won't block!
        self.send_cmd("blocking_stand")

    def power_off(self, *args, **kwargs):
        self.remote_robot_killer.publish(True)
