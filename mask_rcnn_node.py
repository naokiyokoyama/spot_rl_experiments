import time

import cv2
import rospy
from cv_bridge import CvBridge
from mask_rcnn_detectron2.inference import MaskRcnnInference
from sensor_msgs.msg import CompressedImage
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2
from std_msgs.msg import String

WEIGHTS_FILE = "weights/model_0007499.pth"
MASK_RCNN_VIZ_TOPIC = "/mask_rcnn_visualizations"
DETECTIONS_TOPIC = "/mask_rcnn_detections"


class MaskRcnnNode:
    def __init__(self, spot, weights=WEIGHTS_FILE, visualize=True):
        rospy.init_node("mask_rcnn_node")
        self.spot = spot
        self.visualize = visualize
        self.mri = MaskRcnnInference(weights, score_thresh=0.5)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate ROS topic subscribers and publishers
        rospy.Subscriber(
            HAND_RGB_TOPIC,
            CompressedImage,
            self.hand_rgb_cb,
            queue_size=1,
            buff_size=2 ** 24,
        )
        self.viz_pub = rospy.Publisher(
            MASK_RCNN_VIZ_TOPIC, CompressedImage, queue_size=1
        )
        self.det_pub = rospy.Publisher(DETECTIONS_TOPIC, String, queue_size=1)
        self.hand_rgb_img = None

    def hand_rgb_cb(self, msg):
        self.hand_rgb_img = msg

    @staticmethod
    def format_detections(detections):
        detection_str = []
        for det_idx in range(len(detections)):
            class_id = detections.pred_classes[det_idx]
            score = detections.scores[det_idx]
            x1, y1, x2, y2 = detections.pred_boxes[det_idx].tensor.squeeze(0)
            det_attrs = [str(i.item()) for i in [class_id, score, x1, y1, x2, y2]]
            detection_str.append(",".join(det_attrs))
        detection_str = ";".join(detection_str)
        return detection_str

    def publish_detection(self):
        if self.hand_rgb_img is None:
            rospy.loginfo("[mask_rcnn_node]: No RGB image from hand camera...")
            time.sleep(1)
            return
        img = cv2.cvtColor(
            self.cv_bridge.compressed_imgmsg_to_cv2(self.hand_rgb_img, "bgr8"),
            cv2.COLOR_BGR2RGB,
        )
        pred = self.mri.inference(img)

        if len(pred["instances"]) > 0:
            det_str = self.format_detections(pred["instances"])
        else:
            det_str = "None"
        self.det_pub.publish(det_str)

        if self.visualize:
            viz_img = self.mri.visualize_inference(img, pred)
            viz_img_msg = self.cv_bridge.cv2_to_compressed_imgmsg(viz_img)
            self.viz_pub.publish(viz_img_msg)


def main():
    spot = Spot("MaskRCNNInferenceROS")
    mrn = MaskRcnnNode(spot)
    while not rospy.is_shutdown():
        mrn.publish_detection()


if __name__ == "__main__":
    main()
