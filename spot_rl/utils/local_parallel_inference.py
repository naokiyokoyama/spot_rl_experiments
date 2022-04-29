import argparse
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2, scale_depth_img
from std_msgs.msg import Header, String

from spot_rl.spot_ros_node import MAX_DEPTH, MAX_GRIPPER_DEPTH, SpotRosSubscriber
from spot_rl.utils.depth_filter_node import (
    FILTERED_GRIPPER_DEPTH_TOPIC,
    FILTERED_HEAD_DEPTH_TOPIC,
)
from spot_rl.utils.mask_rcnn_node import (
    DETECTIONS_TOPIC,
    MASK_RCNN_VIZ_TOPIC,
    generate_mrcnn_detections,
    get_deblurgan_model,
    get_mrcnn_model,
    pred2string,
)
from spot_rl.utils.stopwatch import Stopwatch
from spot_rl.utils.utils import construct_config

MAX_PUBLISH_FREQ = 20


class SpotLocalObsPublisher:
    sources = []
    publishers = {}
    name = ""

    def __init__(self, spot):
        rospy.init_node(self.name, disable_signals=True)
        self.spot = spot

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate raw image publishers
        self.last_publish = time.time()

        self.pubs = {
            k: rospy.Publisher(k, v, queue_size=1, tcp_nodelay=True)
            for k, v in self.publishers.items()
        }

        rospy.loginfo(f"[{self.name}]: Publisher initialized.")

    def publish_msgs(self):
        st = time.time()
        if st < self.last_publish + 1 / MAX_PUBLISH_FREQ:
            return

        image_responses = self.spot.get_image_responses(self.sources, quality=100)
        imgs = [image_response_to_cv2(r) for r in image_responses]
        imgs = {k: v for k, v in zip(self.sources, imgs)}
        self._publish(imgs)
        self.last_publish = time.time()

    def _publish(self, imgs):
        raise NotImplementedError


class SpotLocalNavObsPublisher(SpotLocalObsPublisher):
    sources = [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
    publishers = {FILTERED_HEAD_DEPTH_TOPIC: Image}
    name = "spot_local_nav_obs_publisher"

    def _publish(self, imgs):
        full_depth = np.hstack(
            [imgs[SpotCamIds.FRONTRIGHT_DEPTH], imgs[SpotCamIds.FRONTLEFT_DEPTH]]
        )
        full_depth = scale_depth_img(full_depth, max_depth=MAX_DEPTH)
        full_depth = np.uint8(full_depth * 255.0)
        filtered_depth = SpotRosSubscriber.filter_depth(full_depth, max_depth=MAX_DEPTH)
        img_msg = self.cv_bridge.cv2_to_imgmsg(filtered_depth, "mono8")
        self.pubs[FILTERED_HEAD_DEPTH_TOPIC].publish(img_msg)


class SpotLocalGazeObsPublisher(SpotLocalObsPublisher):
    sources = [SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME, SpotCamIds.HAND_COLOR]
    publishers = {
        FILTERED_GRIPPER_DEPTH_TOPIC: Image,
        DETECTIONS_TOPIC: String,
        MASK_RCNN_VIZ_TOPIC: Image,
    }
    name = "spot_local_gaze_obs_publisher"

    def __init__(self, spot):
        super().__init__(spot)
        self.config = construct_config()
        self.mrcnn = get_mrcnn_model(self.config)
        self.deblur_gan = get_deblurgan_model(self.config)
        self.image_scale = self.config.IMAGE_SCALE
        rospy.loginfo(f"[{self.name}]: Models loaded.")

    def _publish(self, imgs):
        timestamp = rospy.Time.now()
        stamped_header = Header(stamp=timestamp)

        stopwatch = Stopwatch()

        grip_depth = scale_depth_img(
            imgs[SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME], max_depth=MAX_GRIPPER_DEPTH
        )
        grip_depth = np.uint8(grip_depth * 255.0)
        filtered_depth = SpotRosSubscriber.filter_depth(
            grip_depth, max_depth=MAX_GRIPPER_DEPTH
        )
        depth_msg = self.cv_bridge.cv2_to_imgmsg(filtered_depth, "mono8")
        depth_msg.header = stamped_header
        stopwatch.record("depth_filt")

        # Publish the Mask RCNN detections
        pred, viz_img = generate_mrcnn_detections(
            imgs[SpotCamIds.HAND_COLOR],
            scale=self.image_scale,
            mrcnn=self.mrcnn,
            grayscale=self.config.GRAYSCALE_MASK_RCNN,
            deblurgan=self.deblur_gan,
            return_img=True,
            stopwatch=stopwatch,
        )
        detections_str = f"{int(timestamp.nsecs)}|{pred2string(pred)}"

        viz_img = self.mrcnn.visualize_inference(viz_img, pred)
        stopwatch.print_stats()
        if not detections_str.endswith("None"):
            print(detections_str)
        viz_img_msg = self.cv_bridge.cv2_to_imgmsg(viz_img)
        viz_img_msg.header = stamped_header
        stopwatch.record("vis_secs")

        stopwatch.print_stats()

        self.pubs[FILTERED_GRIPPER_DEPTH_TOPIC].publish(depth_msg)
        self.pubs[DETECTIONS_TOPIC].publish(detections_str)
        self.pubs[MASK_RCNN_VIZ_TOPIC].publish(viz_img_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nav", action="store_true")
    args = parser.parse_args()

    spot = Spot("SpotLocalNavObsPublisher" if args.nav else "SpotLocalGazeObsPublisher")
    n = (SpotLocalNavObsPublisher if args.nav else SpotLocalGazeObsPublisher)(spot)

    while not rospy.is_shutdown():
        n.publish_msgs()
