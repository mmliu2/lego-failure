# test gap classifier and save individual images

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

from gap_detection import lego_face_segmenter


class GapTestNode:
    def __init__(self, robot_name, segmenter):
        self.robot_name = robot_name
        self.bridge = CvBridge()

        self.segmenter = segmenter

        self.now = int(datetime.now().strftime("%Y%m%d%H%M"))
        self.save_dir = f'/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/gap_data_results/saved_gap_images/gen3_{self.now}/'
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

                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                gap_prediction, seg_img, vis_img = self.segmenter.segment(self.image)
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                filename_t = os.path.join(self.save_dir, f"{self.counter:04d}_segmented.jpg")
                cv2.imwrite(filename_t, seg_img)
                filename_t = os.path.join(self.save_dir, f"{self.counter:04d}_vis.jpg")
                cv2.imwrite(filename_t, vis_img)
                
                self.counter += 1
            else:
                print("No image received yet.")


if __name__ == '__main__':
    robot_name = sys.argv[1] 
    assert(robot_name in ['yk_architect', 'yk_builder', 'yk_creator', 'yk_destroyer'])

    rospy.init_node(f'{robot_name}_gap_test_node', anonymous=False)

    dataset_path = "/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/lego-gap-3/train/"
    output_path = "/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/models/segmenter_output"
    segmenter = lego_face_segmenter.LegoFaceSegmenter(dataset_path=dataset_path, output_dir=output_path, train=False)
    print('LegoFaceSegmenter initialized.')

    node = GapTestNode(robot_name, segmenter)

    rospy.spin()