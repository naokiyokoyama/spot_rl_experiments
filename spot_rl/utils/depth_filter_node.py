import argparse
import time

import blosc
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_wrapper.spot import SpotCamIds
from std_msgs.msg import Bool, ByteMultiArray

from spot_rl.spot_ros_node import MAX_DEPTH, MAX_GRIPPER_DEPTH, SpotRosSubscriber

INIT_DEPTH_FILTERING = "/initiate_depth_filtering"
KILL_DEPTH_FILTERING = "/kill_depth_filtering"

COMPRESSED_IMAGES_TOPIC = "/spot_cams/compressed_images"
DETECTIONS_DEPTH_TOPIC = "/mask_rcnn_depth"

FILTERED_HEAD_DEPTH_TOPIC = "/filtered_head_depth_topic"
FILTERED_GRIPPER_DEPTH_TOPIC = "/filtered_gripper_depth_topic"


class DepthFilteringNode:
    def __init__(self, node_name, head_depth=True):
        self.active = True
        self.head_depth = head_depth

        rospy.init_node(node_name, disable_signals=True)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Establish signal and image subscribers
        rospy.Subscriber(
            INIT_DEPTH_FILTERING, Bool, self.init_depth_filtering, queue_size=1
        )
        rospy.Subscriber(
            KILL_DEPTH_FILTERING, Bool, self.kill_depth_filtering, queue_size=1
        )
        if head_depth:
            depth_topic = COMPRESSED_IMAGES_TOPIC
            msg_type = ByteMultiArray
        else:
            depth_topic = DETECTIONS_DEPTH_TOPIC
            msg_type = Image

        rospy.Subscriber(
            depth_topic,
            msg_type,
            self.input_depth_callback,
            queue_size=1,
            buff_size=2 ** 30,
        )

        # Detection topic publisher
        pub_topic = (
            FILTERED_HEAD_DEPTH_TOPIC if head_depth else FILTERED_GRIPPER_DEPTH_TOPIC
        )
        self.filtered_depth_pub = rospy.Publisher(pub_topic, Image, queue_size=1)
        print(node_name, "finished initializing!")

    def init_depth_filtering(self, msg):
        self.active = True

    def kill_depth_filtering(self, msg):
        self.active = False

    def input_depth_callback(self, msg):
        """We process every image that comes at us."""
        if not self.active:
            return

        if self.head_depth:
            msg.layout.dim, timestamp_dim = msg.layout.dim[:-1], msg.layout.dim[-1]
            latency = (
                int(str(int(time.time() * 1000))[-6:]) - timestamp_dim.size
            ) / 1000
            print("Latency: ", latency)
            if latency > 0.5:
                return
            byte_data = (np.array(msg.data) + 128).astype(np.uint8)
            size_and_labels = [
                (int(dim.size), str(dim.label)) for dim in msg.layout.dim
            ]
            start = 0
            eyes = {}
            for size, label in size_and_labels:
                end = start + size
                if "depth" in label:
                    img = blosc.unpack_array(byte_data[start:end].tobytes())
                    if label == SpotCamIds.FRONTLEFT_DEPTH:
                        eyes["left"] = img
                    elif label == SpotCamIds.FRONTRIGHT_DEPTH:
                        eyes["right"] = img
                start += size
            if len(eyes) == 2:
                depth_img = np.hstack([eyes["right"], eyes["left"]])
            else:
                raise RuntimeError("Head depth not found!")
        else:
            depth_img = self.cv_bridge.imgmsg_to_cv2(msg, "mono8")
        max_depth = MAX_DEPTH if self.head_depth else MAX_GRIPPER_DEPTH
        filtered_depth = SpotRosSubscriber.filter_depth(depth_img, max_depth=max_depth)
        filtered_depth_msg = self.cv_bridge.cv2_to_imgmsg(filtered_depth, "mono8")

        if hasattr(msg, "header"):
            filtered_depth_msg.header = msg.header
        self.filtered_depth_pub.publish(filtered_depth_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", action="store_true")
    args = parser.parse_args()

    node_name = f"{'head' if args.head else 'gripper'}_depth_filtering"
    filtering_node = DepthFilteringNode(node_name, head_depth=args.head)
    rospy.spin()
