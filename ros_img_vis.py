import time
from collections import deque

import cv2
import numpy as np
import rospy
from spot_wrapper.utils import resize_to_tallest

from spot_ros_node import SpotRosSubscriber


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
        self.decompress_imgs()
        msgs = [self.front_depth, self.hand_depth, self.hand_rgb, self.det]
        imgs = [i for i in msgs if i is not None]
        imgs = [
            i if i.shape[-1] == 3 else cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in imgs
        ]

        # Make sure all imgs are same height
        img = resize_to_tallest(imgs, hstack=True)
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
