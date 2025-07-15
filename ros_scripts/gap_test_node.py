#!/usr/bin/env python3
import rospy

import numpy as np
from PIL import Image
from cv_bridge import CvBridge
import cv2
from datetime import datetime

from sensor_msgs.msg import CompressedImage

import os
import sys

from gap_detection import gap_detector


class GapTestNode:
    def __init__(self, robot_name, detector):
        self.robot_name = robot_name
        self.bridge = CvBridge()

        self.detector = detector

        self.now = int(datetime.now().strftime("%Y%m%d%H%M"))
        self.save_dir = f'/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/gap_test_results/gen3_{self.now}/'
        os.makedirs(self.save_dir, exist_ok=True)
        self.image = None
        self.counter = 0

        rospy.Subscriber(f'/{robot_name}/gen3_image/compressed', CompressedImage, self.image_callback)
        rospy.loginfo("GapTestNode initialized. Waiting for images...")
        self.run()

    def image_callback(self, msg):
        try:
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def run(self):
        while not rospy.is_shutdown():
            input("Press enter to save image:")
            if self.image is not None:
                filename = os.path.join(self.save_dir, f"{self.counter:04d}.jpg")
                cv2.imwrite(filename, self.image)
                rospy.loginfo(f"Saved image: {filename}")
                filename_t = os.path.join(self.save_dir, f"{self.counter:04d}_result.jpg")
                result, vis_img = self.detector(self.image)
                rospy.loginfo(f"GAP: {result}")
                cv2.imwrite(filename_t, vis_img)
                rospy.loginfo(f"Saved image: {filename_t}")
                self.counter += 1
            else:
                print("No image received yet.")


if __name__ == '__main__':
    robot_name = sys.argv[1] 
    assert(robot_name in ['yk_architect', 'yk_builder', 'yk_creator', 'yk_destroyer'])

    rospy.init_node(f'{robot_name}_gap_test_node', anonymous=False)

    if robot_name == 'yk_destroyer':
        detector = gap_detector.GapDetector(center=(300, 288), size=352, theta=0)
    else:
        detector = gap_detector.GapDetector(center=(468, 450), size=550, theta=0)

    node = GapTestNode(robot_name, detector)

    rospy.spin()