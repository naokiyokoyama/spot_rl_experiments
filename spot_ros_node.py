import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2, scale_depth_img

FILTERED_FRONT_DEPTH_PUB = "/spot_cams/filtered_front_depth"
SPOT_IMAGE_SOURCES = [
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME,
    SpotCamIds.HAND_COLOR,
]
MAX_DEPTH = 3.5
MAX_GRIPPER_DEPTH = 1.7
# MAX_GRIPPER_DEPTH = 10.0


class SpotRosNode:
    def __init__(self, spot):
        rospy.init_node("spot_ros_node")
        self.spot = spot

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate raw image publishers
        self.sources = SPOT_IMAGE_SOURCES
        self.img_pubs = [
            rospy.Publisher(f"/spot_cams/{i}", Image, queue_size=5)
            for i in self.sources
        ]

        # Instantiate filtered image publishers
        self.filter_front_depth = (
            SpotCamIds.FRONTLEFT_DEPTH in self.sources
            and SpotCamIds.FRONTRIGHT_DEPTH in self.sources
        )
        if self.filter_front_depth:
            self.filtered_front_depth_pub = rospy.Publisher(
                FILTERED_FRONT_DEPTH_PUB, Image, queue_size=5
            )

    def publish_msgs(self):
        image_responses = self.spot.get_image_responses(self.sources)
        # Publish raw images
        depth_eyes = {}
        for pub, src, response in zip(self.img_pubs, self.sources, image_responses):
            img = image_response_to_cv2(response)
            if src == SpotCamIds.HAND_COLOR:
                img_type = "bgr8"
            else:  # depth images
                img_type = "mono8"

            # Publish filtered front depth images
            if (
                src in [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
                and self.filter_front_depth
            ):
                depth_eyes[src] = img
                continue
            elif src == SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME:
                img = scale_depth_img(img, max_depth=MAX_GRIPPER_DEPTH)
                img = np.uint8(img * 255.0)

            img_msg = self.cv_bridge.cv2_to_imgmsg(img, img_type)
            pub.publish(img_msg)

        # Filter and publish
        if self.filter_front_depth:
            # Merge
            d_keys = [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
            merged = np.hstack([depth_eyes[d] for d in d_keys])
            # Filter
            merged = self.filter_depth(merged, MAX_DEPTH)
            filtered_msg = self.cv_bridge.cv2_to_imgmsg(merged, "mono8")
            self.filtered_front_depth_pub.publish(filtered_msg)

    @staticmethod
    def filter_depth(img, max_depth):
        img = scale_depth_img(img, max_depth=max_depth)
        img = np.uint8(img * 255.0)
        # Blur
        for _ in range(20):
            filtered = cv2.medianBlur(img, 15)
            filtered[img > 0] = img[img > 0]
            img = filtered

        return img


def main():
    spot = Spot("spot_ros_node")
    srn = SpotRosNode(spot)
    rospy.loginfo("[spot_ros_node]: Publishing has started.")
    while not rospy.is_shutdown():
        srn.publish_msgs()


if __name__ == "__main__":
    main()
