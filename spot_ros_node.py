import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

FILTERED_FRONT_DEPTH_PUB = "/spot_cams/filtered_front_depth"
SPOT_IMAGE_SOURCES = [
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.HAND_DEPTH,
    SpotCamIds.HAND_COLOR,
]


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

    def publish_msgs(self):
        image_responses = self.spot.get_image_responses(self.sources)
        # Publish raw images
        for pub, src, response in zip(self.img_pubs, self.sources, image_responses):
            img = image_response_to_cv2(response)
            if "depth" in src:
                img_type = "mono16"
            elif src == SpotCamIds.HAND_COLOR:
                img_type = "bgr8"
            else:
                img_type = "mono8"
            img_msg = self.cv_bridge.cv2_to_imgmsg(img, img_type)
            pub.publish(img_msg)


def main():
    spot = Spot("spot_ros_node")
    srn = SpotRosNode(spot)
    while not rospy.is_shutdown():
        srn.publish_msgs()


if __name__ == "__main__":
    main()
