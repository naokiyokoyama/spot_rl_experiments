import os.path as osp

import blosc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from deblur_gan.predictor import DeblurGANv2
from mask_rcnn_detectron2.inference import MaskRcnnInference
from sensor_msgs.msg import Image
from spot_wrapper.spot import SpotCamIds
from std_msgs.msg import Bool, ByteMultiArray, Float32, Header, String

from spot_rl.utils.timer import Stopwatch
from spot_rl.utils.utils import construct_config, get_default_parser

COMPRESSED_IMAGES_TOPIC = "/spot_cams/compressed_images"
INIT_MASK_RCNN = "/initiate_mask_rcnn_node"
KILL_MASK_RCNN = "/kill_mask_rcnn_node"

DETECTIONS_TOPIC = "/mask_rcnn_detections"
IMAGE_SCALE_TOPIC = "/mask_rcnn_image_scale"
DETECTIONS_DEPTH_TOPIC = "/mask_rcnn_depth"

MASK_RCNN_VIZ_TOPIC = "/mask_rcnn_visualizations"


class MaskRCnnNode:
    def __init__(self, config, node_name):
        self.config = config
        self.image_scale = config.IMAGE_SCALE
        self.active = True
        self.new_img = False
        self.msg = None

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Load models
        self.mrcnn = get_mrcnn_model(config)
        self.deblur_gan = get_deblurgan_model(config)
        # print("Warming up...")
        # for _ in range(10):
        #     self.run_inference(use_dummy=True)
        # print("Done warming up.")

        rospy.init_node(node_name, disable_signals=True)

        # Detection topic publisher
        self.detection_pub = rospy.Publisher(DETECTIONS_TOPIC, String, queue_size=1)
        self.detection_depth_pub = rospy.Publisher(
            DETECTIONS_DEPTH_TOPIC, Image, queue_size=1
        )
        self.mrcnn_viz_pub = rospy.Publisher(MASK_RCNN_VIZ_TOPIC, Image, queue_size=1)

        # Establish signal and image subscribers
        rospy.Subscriber(
            COMPRESSED_IMAGES_TOPIC,
            ByteMultiArray,
            self.compressed_callback,
            queue_size=1,
            buff_size=2 ** 30,
        )
        rospy.Subscriber(INIT_MASK_RCNN, Bool, self.init_mask_rcnn, queue_size=1)
        rospy.Subscriber(KILL_MASK_RCNN, Bool, self.kill_mask_rcnn, queue_size=1)
        rospy.Subscriber(IMAGE_SCALE_TOPIC, Float32, self.update_scale, queue_size=1)

    def init_mask_rcnn(self, msg):
        self.active = True

    def kill_mask_rcnn(self, msg):
        self.active = False

    def update_scale(self, msg):
        self.image_scale = msg.data

    def compressed_callback(self, msg):
        self.new_img = True
        self.msg = msg
        self.run_inference()

    def uncompress_imgs(self):
        msg = self.msg
        if msg is None:
            return None, None
        """We process every image that comes at us. We delegate filtering of the depth
        image to another node."""
        if not self.active:
            return None, None
        # Uncompress the gripper images
        byte_data = (np.array(msg.data) + 128).astype(np.uint8)
        size_and_labels = [(int(dim.size), str(dim.label)) for dim in msg.layout.dim]
        start = 0
        hand_rgb, hand_depth = None, None

        for size, label in size_and_labels:
            end = start + size
            if "depth" in label:
                img = blosc.unpack_array(byte_data[start:end].tobytes())
            else:
                rgb_bytes = byte_data[start:end]
                img = cv2.imdecode(rgb_bytes, cv2.IMREAD_COLOR)

            if label == SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME:
                hand_depth = img
            elif label == SpotCamIds.HAND_COLOR:
                hand_rgb = img
            start += size

        assert hand_rgb is not None
        assert hand_depth is not None

        return hand_rgb, hand_depth

    def run_inference(self, use_dummy=False):
        if not use_dummy and not self.new_img:
            return
        self.new_img = False

        if use_dummy:
            hand_rgb = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            stopwatch, timestamp, stamped_header = None, None, None
        else:
            stopwatch = Stopwatch()
            timestamp = rospy.Time.now()
            stamped_header = Header(stamp=timestamp)
            hand_rgb, hand_depth = self.uncompress_imgs()
            if hand_depth is None:
                return
            stopwatch.record("uncompress_secs")
            # Send the depth image off to the depth filtering node
            hand_depth_msg = self.cv_bridge.cv2_to_imgmsg(hand_depth, "mono8")
            hand_depth_msg.header = stamped_header
            self.detection_depth_pub.publish(hand_depth_msg)

        # Publish the Mask RCNN detections
        pred, viz_img = generate_mrcnn_detections(
            hand_rgb,
            scale=self.image_scale,
            mrcnn=self.mrcnn,
            grayscale=self.config.GRAYSCALE_MASK_RCNN,
            deblurgan=self.deblur_gan,
            return_img=True,
            stopwatch=stopwatch,
        )
        if not use_dummy:
            detections_str = f"{int(timestamp.nsecs)}|{pred2string(pred)}"
            self.detection_pub.publish(detections_str)

            viz_img = self.mrcnn.visualize_inference(viz_img, pred)
            if not detections_str.endswith("None"):
                print(detections_str)
            viz_img_msg = self.cv_bridge.cv2_to_imgmsg(viz_img)
            viz_img_msg.header = stamped_header
            self.mrcnn_viz_pub.publish(viz_img_msg)
            stopwatch.record("vis_secs")

            stopwatch.print_stats()


def generate_mrcnn_detections(
    img, scale, mrcnn, grayscale=True, deblurgan=None, return_img=False, stopwatch=None
):
    if scale != 1.0:
        img = cv2.resize(
            img,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    if deblurgan is not None:
        img = deblurgan(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if stopwatch is not None:
            stopwatch.record("deblur_secs")
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    detections = mrcnn.inference(img)
    if stopwatch is not None:
        stopwatch.record("mrcnn_secs")

    if return_img:
        return detections, img

    return detections


def pred2string(pred):
    detections = pred["instances"]
    if len(detections) == 0:
        return "None"

    detection_str = []
    for det_idx in range(len(detections)):
        class_id = detections.pred_classes[det_idx]
        score = detections.scores[det_idx]
        x1, y1, x2, y2 = detections.pred_boxes[det_idx].tensor.squeeze(0)
        det_attrs = [str(i.item()) for i in [class_id, score, x1, y1, x2, y2]]
        detection_str.append(",".join(det_attrs))
    detection_str = ";".join(detection_str)
    return detection_str


def get_mrcnn_model(config):
    mask_rcnn_weights = (
        config.WEIGHTS.MRCNN_50 if config.USE_FPN_R50 else config.WEIGHTS.MRCNN
    )
    mask_rcnn_device = config.DEVICE
    config_path = "50" if config.USE_FPN_R50 else "101"
    mrcnn = MaskRcnnInference(
        mask_rcnn_weights,
        score_thresh=0.7,
        device=mask_rcnn_device,
        config_path=config_path,
    )
    return mrcnn


def get_deblurgan_model(config):
    if config.USE_DEBLURGAN and config.USE_MRCNN:
        weights_path = config.WEIGHTS.DEBLURGAN
        model_name = osp.basename(weights_path).split(".")[0]
        print("Loading DeblurGANv2 with:", weights_path)
        model = DeblurGANv2(weights_path=weights_path, model_name=model_name)
        return model
    return None


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("-m", "--use-mixer", action="store_true")
    args = parser.parse_args()
    config = construct_config(args.opts)

    mrcnn_node = MaskRCnnNode(config, "mask_rcnn_node")
    rospy.spin()
    # while True:
    #     mrcnn_node.run_inference()
