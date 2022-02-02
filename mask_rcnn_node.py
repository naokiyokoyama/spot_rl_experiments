import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from mask_rcnn_detectron2.inference import MaskRcnnInference
from sensor_msgs.msg import Image
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2
from std_msgs.msg import String

WEIGHTS_FILE = "/home/naoki/gt/spot/mask_rcnn_detectron2/weights/model_0007499.pth"
VIZ_TOPIC = "/mask_rcnn_visualizations"
DET_TOPIC = "/mask_rcnn_detections"
HAND_COLOR_TOPIC = f"/spot_cams/{SpotCamIds.HAND_COLOR}"


class MaskRcnnNode:
    def __init__(self, spot, weights=WEIGHTS_FILE, visualize=True):
        rospy.init_node("mask_rcnn_node")
        self.spot = spot
        self.visualize = visualize
        self.mri = MaskRcnnInference(weights, score_thresh=0.5)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate ROS topic subscribers and publishers
        rospy.Subscriber(HAND_COLOR_TOPIC, Image, self.hand_color_cb)
        self.viz_pub = rospy.Publisher(VIZ_TOPIC, Image, queue_size=5)
        self.det_pub = rospy.Publisher(DET_TOPIC, String, queue_size=5)
        self.hand_color_img = np.zeros([256, 256, 3], dtype=np.uint8)

    def hand_color_cb(self, msg):
        self.hand_color_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

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
        img = cv2.cvtColor(self.hand_color_img, cv2.COLOR_BGR2RGB)
        pred = self.mri.inference(img)

        if len(pred["instances"]) > 0:
            det_str = self.format_detections(pred["instances"])
        else:
            det_str = "None"
        self.det_pub.publish(det_str)

        if self.visualize:
            viz_img = self.mri.visualize_inference(img, pred)
            viz_img_msg = self.cv_bridge.cv2_to_imgmsg(viz_img, "bgr8")
            self.viz_pub.publish(viz_img_msg)


def main():
    spot = Spot("MaskRCNNInferenceROS")
    mrn = MaskRcnnNode(spot)
    while not rospy.is_shutdown():
        mrn.publish_detection()


if __name__ == "__main__":
    main()
