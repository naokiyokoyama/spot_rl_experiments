import time
from collections import deque

import cv2
import numpy as np
import rospy
from spot_wrapper.utils import resize_to_tallest

from spot_rl.spot_ros_node import SpotRosSubscriber


class SpotRosVisualizer(SpotRosSubscriber):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.last_render = time.time()
        self.fps_buffer = deque(maxlen=10)

    def vis_imgs(self):
        # Skip if no messages were updated
        if not self.updated:
            return

        # Gather latest images
        self.uncompress_imgs()
        orig_msgs = [self.front_depth, self.hand_depth[:, 124:-60], self.hand_rgb]
        processed_msgs = [
            self.filter_depth(self.front_depth, max_depth=3.5, whiten_black=True),
            self.filter_depth(
                self.hand_depth[:, 124:-60], max_depth=1.6, whiten_black=True
            ),
            np.ones_like(self.hand_rgb)
            if self.det is None
            else self.cv_bridge.compressed_imgmsg_to_cv2(self.det),
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
        cv2.imshow("ROS Spot Images", img)
        cv2.waitKey(1)

        self.updated = False
        self.fps_buffer.append(1 / (time.time() - self.last_render))
        rospy.loginfo(
            f"{np.mean(self.fps_buffer):.2f} FPS (window size: {len(self.fps_buffer)})"
        )
        self.last_render = time.time()


def main():
    try:
        srv = SpotRosVisualizer("spot_ros_vis_node")
        while not rospy.is_shutdown():
            srv.vis_imgs()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
