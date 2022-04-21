import argparse
import time

import blosc
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2, scale_depth_img
from spot_wrapper.utils import say
from std_msgs.msg import (
    ByteMultiArray,
    Float32MultiArray,
    MultiArrayDimension,
    MultiArrayLayout,
    String,
)

from spot_rl.utils.depth_map_utils import fill_in_multiscale

ROBOT_VEL_TOPIC = "/spot_cmd_velocities"
MASK_RCNN_VIZ_TOPIC = "/mask_rcnn_visualizations"
COMPRESSED_IMAGES_TOPIC = "/spot_cams/compressed_images"
ROBOT_STATE_TOPIC = "/robot_state"
TEXT_TO_SPEECH_TOPIC = "/text_to_speech"
SRC2MSG = {
    SpotCamIds.FRONTLEFT_DEPTH: ByteMultiArray,
    SpotCamIds.FRONTRIGHT_DEPTH: ByteMultiArray,
    SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME: ByteMultiArray,
    SpotCamIds.HAND_COLOR: CompressedImage,
}
MAX_DEPTH = 3.5
MAX_GRIPPER_DEPTH = 1.7

NAV_POSE_BUFFER_LEN = 3


class SpotRosPublisher:
    def __init__(self, spot):
        rospy.init_node("spot_ros_node", disable_signals=True)
        self.spot = spot

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate raw image publishers
        self.sources = list(SRC2MSG.keys())
        self.img_pub = rospy.Publisher(
            COMPRESSED_IMAGES_TOPIC, ByteMultiArray, queue_size=1
        )

        self.last_publish = time.time()
        rospy.loginfo("[spot_ros_node]: Publishing has started.")

    def publish_msgs(self):
        st = time.time()
        image_responses = self.spot.get_image_responses(self.sources, quality=100)
        retrieval_time = time.time() - st
        # Publish raw images
        src2details = {}
        for src, response in zip(self.sources, image_responses):
            img = image_response_to_cv2(response)

            if "depth" in src:
                # Rescale depth images here
                if src == SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME:
                    max_depth = MAX_GRIPPER_DEPTH
                else:
                    max_depth = MAX_DEPTH
                img = scale_depth_img(img, max_depth=max_depth)
                img = np.uint8(img * 255.0)
                img_bytes = blosc.pack_array(
                    img, cname="zstd", clevel=3, shuffle=blosc.NOSHUFFLE
                )
            else:
                # RGB should be JPEG compressed instead of using blosc
                img_bytes = np.array(cv2.imencode(".jpg", img)[1])
                img_bytes = (img_bytes.astype(int) - 128).astype(np.int8)
            src2details[src] = {
                "dims": MultiArrayDimension(label=src, size=len(img_bytes)),
                "bytes": img_bytes,
            }

        depth_bytes = b""
        rgb_bytes = []
        depth_dims = []
        rgb_dims = []
        for k, v in src2details.items():
            if "depth" in k:
                depth_bytes += v["bytes"]
                depth_dims.append(v["dims"])
            else:
                rgb_bytes.append(v["bytes"])
                rgb_dims.append(v["dims"])
        depth_bytes = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth_bytes = depth_bytes.astype(int) - 128
        bytes_data = np.concatenate([depth_bytes, *rgb_bytes])
        dims = depth_dims + rgb_dims

        msg = ByteMultiArray(layout=MultiArrayLayout(dim=dims), data=bytes_data)
        self.img_pub.publish(msg)

        rospy.loginfo(
            f"[spot_ros_node]: Image retrieval / publish time: "
            f"{1 / retrieval_time:.4f} / {1 / (time.time() - self.last_publish):.4f} Hz"
        )
        self.last_publish = time.time()


class SpotRosSubscriber:
    def __init__(self, node_name):
        rospy.init_node(node_name, disable_signals=True)

        # For generating Image ROS msgs
        self.cv_bridge = CvBridge()

        # Instantiate subscribers
        rospy.Subscriber(
            COMPRESSED_IMAGES_TOPIC,
            ByteMultiArray,
            self.compressed_callback,
            queue_size=1,
            buff_size=2 ** 30,
        )
        rospy.Subscriber(
            MASK_RCNN_VIZ_TOPIC,
            CompressedImage,
            self.viz_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.Subscriber(
            ROBOT_STATE_TOPIC,
            Float32MultiArray,
            self.robot_state_callback,
            queue_size=1,
        )

        # Conversion between CompressedImage and cv2
        self.cv_bridge = CvBridge()

        # Msg holders
        self.compressed_imgs_msg = None
        self.front_depth = None
        self.hand_depth = None
        self.hand_rgb = None
        self.det = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.current_arm_pose = None
        self.lock = False

        self.updated = False
        rospy.loginfo(f"[{node_name}]: Subscribing has started.")

    def compressed_callback(self, msg):
        if self.lock:
            return
        self.compressed_imgs_msg = msg
        self.updated = True

    def uncompress_imgs(self):
        assert self.compressed_imgs_msg is not None, "No compressed imgs received!"
        self.lock = True
        byte_data = (np.array(self.compressed_imgs_msg.data) + 128).astype(np.uint8)
        size_and_labels = [
            (int(dim.size), str(dim.label))
            for dim in self.compressed_imgs_msg.layout.dim
        ]
        self.lock = False
        start = 0
        eyes = {}
        for size, label in size_and_labels:
            end = start + size
            if "depth" in label:
                img = blosc.unpack_array(byte_data[start:end].tobytes())
            else:
                rgb_bytes = byte_data[start:end]
                img = cv2.imdecode(rgb_bytes, cv2.IMREAD_COLOR)
            if label == SpotCamIds.FRONTLEFT_DEPTH:
                eyes["left"] = img
            elif label == SpotCamIds.FRONTRIGHT_DEPTH:
                eyes["right"] = img
            elif label == SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME:
                self.hand_depth = img
            elif label == SpotCamIds.HAND_COLOR:
                self.hand_rgb = img
            start += size

        if len(eyes) == 2:
            self.front_depth = np.hstack([eyes["right"], eyes["left"]])

    def viz_callback(self, msg):
        self.det = msg
        self.updated = True

    def robot_state_callback(self, msg):
        self.x, self.y, self.yaw = msg.data[:3]
        self.current_arm_pose = msg.data[3:]

    @staticmethod
    def filter_depth(depth_img, max_depth, whiten_black=True):
        filtered_depth_img = (
            fill_in_multiscale(depth_img.astype(np.float32) * (max_depth / 255.0))[0]
            * (255.0 / max_depth)
        ).astype(np.uint8)
        # Recover pixels that weren't black before but were turned black by filtering
        recovery_pixels = np.logical_and(depth_img != 0, filtered_depth_img == 0)
        filtered_depth_img[recovery_pixels] = depth_img[recovery_pixels]
        if whiten_black:
            filtered_depth_img[filtered_depth_img == 0] = 255
        return filtered_depth_img


class SpotRosProprioceptionPublisher:
    def __init__(self, spot):
        rospy.init_node("spot_ros_proprioception_node", disable_signals=True)
        self.spot = spot

        # Instantiate filtered image publishers
        self.pub = rospy.Publisher(ROBOT_STATE_TOPIC, Float32MultiArray, queue_size=1)
        self.last_publish = time.time()
        rospy.loginfo("[spot_ros_proprioception_node]: Publishing has started.")

        self.nav_pose_buff = None
        self.buff_idx = 0

    def publish_msgs(self):
        st = time.time()
        robot_state = self.spot.get_robot_state()
        msg = Float32MultiArray()
        xy_yaw = self.spot.get_xy_yaw(robot_state=robot_state, use_boot_origin=True)
        if self.nav_pose_buff is None:
            self.nav_pose_buff = np.tile(xy_yaw, [NAV_POSE_BUFFER_LEN, 1])
        else:
            self.nav_pose_buff[self.buff_idx] = xy_yaw
        self.buff_idx = (self.buff_idx + 1) % NAV_POSE_BUFFER_LEN
        xy_yaw = np.mean(self.nav_pose_buff, axis=0)

        joints = self.spot.get_arm_proprioception(robot_state=robot_state).values()
        msg.data = np.array(
            list(xy_yaw) + [j.position.value for j in joints],
            dtype=np.float32,
        )

        # Limit publishing to 60 Hz max
        if time.time() - self.last_publish > 1 / 60:
            self.pub.publish(msg)
            rospy.loginfo(
                f"[spot_ros_proprioception_node]: "
                "Proprioception retrieval / publish time: "
                f"{1/(time.time() - st):.4f} / "
                f"{1/(time.time() - self.last_publish):.4f} Hz"
            )
            self.last_publish = time.time()


class SpotVelocitySubscriber:
    def __init__(self, spot):
        self.spot = spot
        rospy.init_node("spot_ros_velocity_node", disable_signals=True)
        rospy.Subscriber(
            ROBOT_VEL_TOPIC, Float32MultiArray, self.vel_callback, queue_size=1
        )
        rospy.loginfo("[spot_ros_velocity_node]: Listening for velocity commands.")
        rospy.spin()

    def vel_callback(self, msg):
        lin_vel, hor_vel, ang_vel, duration = msg.data
        self.spot.set_base_velocity(lin_vel, hor_vel, ang_vel, duration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proprioception", action="store_true")
    parser.add_argument("-t", "--text-to-speech", action="store_true")
    parser.add_argument("-v", "--velocity", action="store_true")
    args = parser.parse_args()

    if args.text_to_speech:
        tts_callback = lambda msg: say(msg.data)
        rospy.init_node("spot_ros_tts_node", disable_signals=True)
        rospy.Subscriber(TEXT_TO_SPEECH_TOPIC, String, tts_callback, queue_size=1)
        rospy.loginfo("[spot_ros_tts_node]: Listening for text to dictate.")
        rospy.spin()
    else:
        if args.proprioception:
            name = "spot_ros_proprioception_node"
            cls = SpotRosProprioceptionPublisher
        elif args.velocity:
            name = "spot_ros_velocity_node"
            cls = SpotVelocitySubscriber
        else:
            name = "spot_ros_node"
            cls = SpotRosPublisher

        spot = Spot(name)
        srn = cls(spot)
        if not args.velocity:
            while not rospy.is_shutdown():
                srn.publish_msgs()


if __name__ == "__main__":
    main()
