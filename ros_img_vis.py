import time
from collections import deque

import blosc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from spot_wrapper.spot import SpotCamIds
from std_msgs.msg import ByteMultiArray

FRONT_DEPTH_TOPIC = "/spot_cams/filtered_front_depth"
HAND_DEPTH_TOPIC = f"/spot_cams/{SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME}"
HAND_RGB_TOPIC = f"/spot_cams/{SpotCamIds.HAND_COLOR}"
DET_TOPIC = "/mask_rcnn_detections"


class RosVisNode:
    def __init__(self):
        rospy.init_node("ros_vis_node", disable_signals=True)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate subscribers
        rospy.Subscriber(FRONT_DEPTH_TOPIC, ByteMultiArray, self.front_depth_callback)
        rospy.Subscriber(HAND_DEPTH_TOPIC, ByteMultiArray, self.hand_depth_callback)
        rospy.Subscriber(HAND_RGB_TOPIC, CompressedImage, self.hand_rgb_callback)
        rospy.Subscriber(DET_TOPIC, CompressedImage, self.det_callback)

        # Conversion between CompressedImage and Cv2
        self.cv_bridge = CvBridge()

        # Image holders
        self.front_depth = None
        self.hand_depth = None
        self.hand_rgb = None
        self.det = None
        self.update = False
        self.last_render = time.time()

        self.fps_buffer = deque(maxlen=10)

    def front_depth_callback(self, msg):
        self.front_depth = msg
        self.update = True

    def hand_depth_callback(self, msg):
        self.hand_depth = msg
        self.update = True

    def hand_rgb_callback(self, msg):
        self.hand_rgb = msg
        self.update = True

    def det_callback(self, msg):
        self.det = msg
        self.update = True

    def vis_imgs(self):
        # Skip if no messages were updated
        if not self.update:
            return

        msgs = {
            "front_depth": self.front_depth,
            "hand_depth": self.hand_depth,
            "hand_rgb": self.hand_rgb,
            "det": self.det,
        }

        # Gather latest images
        imgs = []
        for k, msg in msgs.items():
            if msg is None:
                continue
            if k in ["front_depth", "hand_depth"]:
                byte_data = msg.data
                byte_data = [(i + 128).to_bytes(1, "big") for i in byte_data]
                mono_img = blosc.unpack_array(b"".join(byte_data))
                imgs.append(cv2.cvtColor(mono_img, cv2.COLOR_GRAY2BGR))
            elif k in ["hand_rgb", "det"]:
                imgs.append(self.cv_bridge.compressed_imgmsg_to_cv2(msg))

        # Make sure all imgs are same height
        tallest = max([i.shape[0] for i in imgs])
        for idx, i in enumerate(imgs):
            height, width = i.shape[:2]
            if height != tallest:
                new_width = int(width * (tallest / height))
                imgs[idx] = cv2.resize(i, (new_width, tallest))

        # Show the images
        img = np.hstack(imgs)
        cv2.imshow("ros_vis_node", img)
        cv2.waitKey(1)

        self.update = False
        self.fps_buffer.append(1 / (time.time() - self.last_render))
        print(np.mean(self.fps_buffer), "FPS")
        self.last_render = time.time()


def main():
    try:
        rvn = RosVisNode()
        rospy.loginfo("[ros_vis_node]: Subscribing has started.")
        while not rospy.is_shutdown():
            rvn.vis_imgs()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
