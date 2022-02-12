import blosc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2, scale_depth_img
from std_msgs.msg import ByteMultiArray

FILTERED_FRONT_DEPTH_PUB = "/spot_cams/filtered_front_depth"
SRC2MSG = {
    SpotCamIds.FRONTLEFT_DEPTH: ByteMultiArray,
    SpotCamIds.FRONTRIGHT_DEPTH: ByteMultiArray,
    SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME: ByteMultiArray,
    SpotCamIds.HAND_COLOR: CompressedImage,
}
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
        self.sources = list(SRC2MSG.keys())
        self.img_pubs = [
            rospy.Publisher(f"/spot_cams/{k}", v, queue_size=5)
            for k, v in SRC2MSG.items()
        ]

        # Instantiate filtered image publishers
        self.filter_front_depth = (
            SpotCamIds.FRONTLEFT_DEPTH in self.sources
            and SpotCamIds.FRONTRIGHT_DEPTH in self.sources
        )
        if self.filter_front_depth:
            self.filtered_front_depth_pub = rospy.Publisher(
                FILTERED_FRONT_DEPTH_PUB, ByteMultiArray, queue_size=5
            )

    def publish_msgs(self):
        image_responses = self.spot.get_image_responses(self.sources)
        # Publish raw images
        depth_eyes = {}
        for pub, src, response in zip(self.img_pubs, self.sources, image_responses):
            img = image_response_to_cv2(response)

            # Publish filtered front depth images later
            if (
                src in [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
                and self.filter_front_depth
            ):
                depth_eyes[src] = img
                continue
            elif src == SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME:
                img = scale_depth_img(img, max_depth=MAX_GRIPPER_DEPTH)
                img = np.uint8(img * 255.0)

            if src == SpotCamIds.HAND_COLOR:
                msg = self.cv_bridge.cv2_to_compressed_imgmsg(img)
            else:
                msg = blosc.pack_array(
                    img, cname="zstd", clevel=1, shuffle=blosc.NOSHUFFLE
                )
                msg = ByteMultiArray(data=[i - 128 for i in msg])

            pub.publish(msg)

        # Filter and publish
        if self.filter_front_depth:
            # Merge
            d_keys = [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
            merged = np.hstack([depth_eyes[d] for d in d_keys])
            # Filter
            merged = self.filter_depth(merged, MAX_DEPTH)
            msg = blosc.pack_array(
                merged, cname="zstd", clevel=1, shuffle=blosc.NOSHUFFLE
            )
            msg = ByteMultiArray(data=[i - 128 for i in msg])
            self.filtered_front_depth_pub.publish(msg)

    @staticmethod
    def filter_depth(img, max_depth):
        img = scale_depth_img(img, max_depth=max_depth)
        img = np.uint8(img * 255.0)
        # Blur
        for _ in range(5):
            filtered = cv2.medianBlur(img, 19)
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
