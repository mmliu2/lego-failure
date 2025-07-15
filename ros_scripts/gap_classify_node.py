#!/usr/bin/env python3
import rospy

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

import sys
import shutil
import os

from gap_detection import gap_detector

class GapClassifyNode:
    def __init__(self, robot_name, detector):
        self.robot_name = robot_name
        self.detector = detector

        self.rate = rospy.Rate(1)
        self.latest_image = None
        self.frame_count = 0
        self.save_dir = '../data/gap_classify_node_visualization'
        shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()

        self.result_pub = rospy.Publisher(f'/{robot_name}/gap_classify', Bool, queue_size=1)
        rospy.Subscriber(f'/{robot_name}/gen3_image/compressed', CompressedImage, self.image_callback)
        # self.vis_pub = rospy.Publisher(f'/{robot_name}/gap_classify_vis', CompressedImage, queue_size=1)
        
        rospy.loginfo("GapClassifyNode initialized. Waiting for images...")

        while not rospy.is_shutdown():
            if self.latest_image is not None:
                self.process_image()
            self.rate.sleep()

    def image_callback(self, msg):
        self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def process_image(self):
        img = self.latest_image

        result = Bool()
        # result_vis = CompressedImage()

        gap_prediction, vis_img = self.detector(img)
        result.data = gap_prediction
        # result_vis.data = vis_img

        self.result_pub.publish(result)
        cv2.imwrite(self.save_dir + f'/{self.frame_count:04}.png', vis_img)
        self.frame_count += 1
        # self.vis_pub.publish(result_vis)

        rospy.loginfo(f"{robot_name} result: %s", result.data)
        

if __name__ == '__main__':
    robot_name = sys.argv[1] 
    assert(robot_name in ['yk_architect', 'yk_builder', 'yk_creator', 'yk_destroyer'])

    if robot_name == 'yk_destroyer':
        detector = gap_detector.GapDetector(center=(300, 288), size=352, theta=0)
    else:
        detector = gap_detector.GapDetector(center=(300, 288), size=352, theta=0)
        
    rospy.init_node(f'{robot_name}_gap_classify_node', anonymous=False)
    node = GapClassifyNode(robot_name, detector)
    rospy.spin()