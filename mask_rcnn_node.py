from mask_rcnn_detectron2.inference import MaskRcnnInference
from bd_spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    image_response_to_cv2,
)
import cv2
from cv_bridge import CvBridge
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

WEIGHTS_FILE = "/home/naoki/gt/spot/mask_rcnn_detectron2/weights/model_0007499.pth"
VIZ_TOPIC = "/mask_rcnn_visualizations"
DET_TOPIC = "/mask_rcnn_detections"


class MaskRcnnNode:
    def __init__(self, spot, weights=WEIGHTS_FILE, visualize=True):
        rospy.init_node("mask_rcnn_node")
        self.spot = spot
        self.visualize = visualize 
        self.mri = MaskRcnnInference(weights, score_thresh=0.5)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate ROS topic publishers
        self.viz_pub = rospy.Publisher(VIZ_TOPIC, Image, queue_size=5)
        self.det_pub = rospy.Publisher(DET_TOPIC, String, queue_size=5)

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
        image_responses = self.spot.get_image_responses([SpotCamIds.HAND_COLOR])
        img = image_response_to_cv2(image_responses[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

if __name__ == '__main__':
    main()