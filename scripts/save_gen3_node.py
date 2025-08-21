#!/usr/bin/env python3
import rospy

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

import argparse
import os
from datetime import datetime

class PickPlaceSaveNode:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.bridge = CvBridge()

        self.now = int(datetime.now().strftime("%Y%m%d%H%M"))
        self.save_dir = f'/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/gen3_saved_images'
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
            s = input("Press enter to save image (-1 to exit): ")
            if s == '-1': exit(0)

            if self.image is not None:
                filename = os.path.join(self.save_dir, f"gen3_{self.now}_{self.counter:04d}.jpg")
                cv2.imwrite(filename, self.image)
                rospy.loginfo(f"Saved image: {filename}")
                self.counter += 1
            else:
                print("No image received yet.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_name', help='Robot to use: yk_architect, yk_builder, yk_creator, yk_destroyer')
    args = parser.parse_args()

    rospy.init_node(f'{args.robot_name}_pick_place_save_node', anonymous=False)
    node = PickPlaceSaveNode(args.robot_name)
    rospy.spin()