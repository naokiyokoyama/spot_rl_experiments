import argparse
import time
from collections import deque

import cv2
import numpy as np
import rospy
import tqdm
from spot_wrapper.utils import resize_to_tallest

from spot_rl.spot_ros_node import SpotRosSubscriber

FOUR_CC = cv2.VideoWriter_fourcc(*"MP4V")
FPS = 30


class SpotRosVisualizer(SpotRosSubscriber):
    def __init__(self, node_name, headless=False):
        super().__init__(node_name + "_" + str(int(time.time())))
        self.last_render = time.time()
        self.fps_buffer = deque(maxlen=10)
        self.recording = False
        self.frames = []
        self.headless = headless
        self.curr_video_time = time.time()
        self.video = True

    def generate_composite(self):
        # Gather latest images
        self.uncompress_imgs()
        orig_msgs = [self.front_depth, self.hand_depth[:, 124:-60], self.hand_rgb]
        processed_msgs = [
            self.filter_depth(self.front_depth, max_depth=3.5, whiten_black=True),
            self.filter_depth(
                self.hand_depth[:, 124:-60], max_depth=1.6, whiten_black=True
            ),
            np.zeros_like(self.hand_rgb)
            if self.det is None
            else self.cv_bridge.imgmsg_to_cv2(self.det),
        ]
        img_rows = []
        for msgs in [orig_msgs, processed_msgs]:
            imgs = [i for i in msgs if i is not None]
            imgs = [
                i if i.shape[-1] == 3 else cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)
                for i in imgs
            ]

            # Make sure all imgs are same height
            img_rows.append(resize_to_tallest(imgs, hstack=True))
        img = np.vstack(img_rows)

        return img

    @staticmethod
    def overlay_text(img, text):
        viz_img = img.copy()
        line, font, font_size, font_thickness = (
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            4,
        )
        text_width, text_height = cv2.getTextSize(
            line, font, font_size, font_thickness
        )[0][:2]
        height, width = img.shape[:2]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        cv2.putText(
            viz_img,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
        return viz_img

    def vis_imgs(self):
        # Skip if no messages were updated
        if self.updated and self.compressed_imgs_msg is not None:
            img = self.generate_composite()
            if not self.headless:
                if not self.recording:
                    cv2.imshow("ROS Spot Images", img)
                else:
                    viz_img = self.overlay_text(img, "RECORDING IS ON!")
                    cv2.imshow("ROS Spot Images", viz_img)
            if self.recording:
                self.frames.append([img.copy(), time.time()])

            self.updated = False
            self.fps_buffer.append(1 / (time.time() - self.last_render))
            mean_fps = np.mean(self.fps_buffer)
            rospy.loginfo(f"{mean_fps:.2f} FPS (window size: {len(self.fps_buffer)})")
            self.last_render = time.time()

        # Logic for recording video
        if not self.headless:
            key = cv2.waitKey(1)
            if key != -1 and ord("r") == key:
                self.recording = not self.recording
            if not self.recording and len(self.frames) > 0:
                self.save_video()

        if self.recording:
            if self.video is None:
                self.video = cv2.VideoWriter(out_path, FOUR_CC, FPS, (width, height))

    def save_video(self):
        # Save the video
        first_frame, first_timestamp = self.frames[0]
        viz_img = self.overlay_text(np.zeros_like(first_frame), "SAVING VIDEO...")
        if not self.headless:
            cv2.imshow("ROS Spot Images", viz_img)
            cv2.waitKey(1)
        height, width = first_frame.shape[:2]
        out_path = f"{time.time()}.mp4"
        video = cv2.VideoWriter(out_path, FOUR_CC, FPS, (width, height))
        curr_time = first_timestamp
        for idx, (frame, timestamp) in enumerate(tqdm.tqdm(self.frames)):
            if idx + 1 >= len(self.frames):
                video.write(frame)
            else:
                next_timestamp = self.frames[idx + 1][1]
                while curr_time < next_timestamp:
                    video.write(frame)
                    curr_time += 1 / FPS
        video.release()
        print(f"Saved video to {out_path}!")
        self.frames = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    srv = None
    try:
        srv = SpotRosVisualizer("spot_ros_vis_node", headless=args.headless)
        if args.record:
            srv.recording = True
        while not rospy.is_shutdown():
            srv.vis_imgs()
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        if srv is not None and srv.frames:
            srv.save_video()


if __name__ == "__main__":
    main()
