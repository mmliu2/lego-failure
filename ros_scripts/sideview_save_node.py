#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

import argparse
import os
from datetime import datetime

class SideviewSaveNode:
    def __init__(self, camera_name, save_depth):
        self.robot_name = camera_name
        self.bridge = CvBridge()

        self.now = int(datetime.now().strftime("%Y%m%d%H%M"))
        self.save_dir = f'/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/brick_drop_depth_data'
        self.color_image = None
        self.depth_image = None
        self.save_depth = save_depth
        self.counter = 0

        rospy.Subscriber(f'/{camera_name}/color/image_raw/compressed', CompressedImage, self.image_color_callback)
        rospy.Subscriber(f'/{camera_name}/depth/image_raw/compressedDepth', CompressedImage, self.image_depth_callback)
        rospy.loginfo("SideviewSaveNode initialized. Waiting for images...")
        self.run()

    def image_color_callback(self, msg):
        try:
            self.color_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def image_depth_callback(self, msg):
        if self.save_depth:
            try:
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                print(np_arr.shape)
                self.depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                print("image_depth", self.depth_image)
            except Exception as e:
                rospy.logerr(f"CV Bridge error: {e}")

    def run(self):
        while not rospy.is_shutdown():
            s = input("Press enter to save image (-1 to exit): ")
            if s == '-1': exit(0)
            
            if self.color_image is not None:
                filename = os.path.join(self.save_dir, f"sideview_{self.now}_{self.counter:04d}.png")
                cv2.imwrite(filename, self.color_image)
                rospy.loginfo(f"Saved color image: {filename}")
            else:
                rospy.loginfo(f"Waiting for color image.")

            if self.depth_image is not None:
                filename = os.path.join(self.save_dir, f"sideview_{self.now}_{self.counter:04d}_depth.png")
                cv2.imwrite(filename, self.depth_image/255)
                rospy.loginfo(f"Saved depth image: {filename}")
            else:
                rospy.loginfo(f"Waiting for depth image.")

            self.counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera selection script.')
    parser.add_argument('--cam_name', help='Camera to use: cam_architect, cam_builder, cam_creator, cam_destroyer')
    parser.add_argument('--depth', action='store_true', help='Save depth image')
    args = parser.parse_args()

    rospy.init_node(f'{args.cam_name}_sideview_save_node', anonymous=False)
    node = SideviewSaveNode(args.cam_name, args.depth)
    rospy.spin()