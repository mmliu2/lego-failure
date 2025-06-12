#!/usr/bin/env python3
import rospy

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import pickle
from cv_bridge import CvBridge
import cv2
from datetime import datetime

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

import os
import sys

class PickPlaceSaveNode:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.bridge = CvBridge()

        self.now = int(datetime.now().strftime("%Y%m%d%H%M"))
        self.save_dir = f'/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/new_pick_place_images'
        self.image = None
        self.counter = 0

        rospy.Subscriber(f'/{robot_name}/gen3_image/compressed', CompressedImage, self.image_callback)
        rospy.loginfo("PickPlaceSaveNode initialized. Waiting for images...")
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
                filename = os.path.join(self.save_dir, f"gen3_{self.now}_{self.counter:04d}.jpg")
                cv2.imwrite(filename, self.image)
                rospy.loginfo(f"Saved image: {filename}")
                self.counter += 1
            else:
                print("No image received yet.")


if __name__ == '__main__':
    robot_name = sys.argv[1] 
    assert(robot_name in ['yk_architect', 'yk_builder', 'yk_creator', 'yk_destroyer'])

    rospy.init_node(f'{robot_name}_pick_place_save_node', anonymous=False)
    node = PickPlaceSaveNode(robot_name)
    rospy.spin()