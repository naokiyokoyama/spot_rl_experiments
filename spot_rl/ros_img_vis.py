import argparse
import os
import os.path as osp
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
        super().__init__(node_name + "_" + str(int(time.time())), proprioception=False)
        self.last_render = time.time()
        self.fps_buffer = deque(maxlen=10)
        self.recording = False
        self.frames = []
        self.headless = headless
        self.curr_video_time = time.time()
        self.out_path = None
        self.video = None
        self.lock = False
        self.dim = None
        self.new_video_started = False

        while self.compressed_imgs_msg is None:
            pass

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
    def overlay_text(img, text, color=(0, 0, 255)):
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
            color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )
        return viz_img

    def vis_imgs(self):
        # Skip if no messages were updated
        currently_saving = not self.recording and self.frames
        self.lock = currently_saving
        img = self.generate_composite() if not currently_saving else None
        if not self.headless:
            if img is not None:
                if self.recording:
                    viz_img = self.overlay_text(img, "RECORDING IS ON!")
                    cv2.imshow("ROS Spot Images", viz_img)
                else:
                    cv2.imshow("ROS Spot Images", img)

            key = cv2.waitKey(1)
            if key != -1:
                if ord("r") == key and not currently_saving:
                    self.recording = not self.recording
                elif ord("q") == key:
                    exit()

        if img is not None:
            self.dim = img.shape[:2]

            # FPS metrics
            self.fps_buffer.append(1 / (time.time() - self.last_render))
            mean_fps = np.mean(self.fps_buffer)
            rospy.loginfo(f"{mean_fps:.2f} FPS (window size: {len(self.fps_buffer)})")
            self.last_render = time.time()

            # Video recording
            if self.recording:
                self.frames.append(time.time())
                if self.video is None:
                    height, width = img.shape[:2]
                    self.out_path = f"{time.time()}.mp4"
                    self.video = cv2.VideoWriter(
                        self.out_path, FOUR_CC, FPS, (width, height)
                    )
                self.video.write(img)

        if currently_saving and not self.recording:
            self.save_video()

    def save_video(self):
        if self.video is None:
            return
        # Close window while we work
        cv2.destroyAllWindows()

        # Save current buffer
        self.video.release()
        old_video = cv2.VideoCapture(self.out_path)
        ret, img = old_video.read()

        # Re-make video with correct timing
        height, width = self.dim
        self.new_video_started = True
        new_video = cv2.VideoWriter(
            self.out_path.replace(".mp4", "_final.mp4"),
            FOUR_CC,
            FPS,
            (width, height),
        )
        curr_video_time = self.frames[0]
        for idx, timestamp in enumerate(tqdm.tqdm(self.frames)):
            if not ret:
                break
            if idx + 1 >= len(self.frames):
                new_video.write(img)
            else:
                next_timestamp = self.frames[idx + 1]
                while curr_video_time < next_timestamp:
                    new_video.write(img)
                    curr_video_time += 1 / FPS
            ret, img = old_video.read()

        new_video.release()
        os.remove(self.out_path)
        self.video, self.out_path, self.frames = None, None, []
        self.new_video_started = False

    def delete_videos(self):
        for i in [self.out_path, self.out_path.replace(".mp4", "_final.mp4")]:
            if osp.isfile(i):
                os.remove(i)


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
    except Exception as e:
        print("Ending script.")
        if not args.headless:
            cv2.destroyAllWindows()
        if srv is not None:
            try:
                if srv.new_video_started:
                    print("Deleting unfinished videos.")
                    srv.delete_videos()
                else:
                    srv.save_video()
            except:
                print("Deleting unfinished videos")
                srv.delete_videos()
                exit()
        raise e


if __name__ == "__main__":
    main()
