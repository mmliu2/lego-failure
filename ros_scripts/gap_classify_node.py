#!/usr/bin/env python3
import rospy

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool

import sys
import shutil
import os

from gap_detection import gap_detector

class GapClassifyNode:
    def __init__(self, robot_name, detector, save=False):
        self.robot_name = robot_name
        self.detector = detector

        self.rate = rospy.Rate(2)
        self.latest_image = None
        self.frame_count = 0

        self.save_dir = '/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/gap_classify_node_visualization'
        shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.save = save

        self.bridge = CvBridge()

        self.result_pub = rospy.Publisher(f'/{robot_name}/gap_classify', Bool, queue_size=1)
        rospy.Subscriber(f'/{robot_name}/gen3_image/compressed', CompressedImage, self.image_callback)
        self.vis_pub = rospy.Publisher(f'/{robot_name}/gap_classify_vis', Image, queue_size=1)
        self.edge_pub = rospy.Publisher(f'/{robot_name}/gap_classify_edge', Image, queue_size=1)
        
        rospy.loginfo("GapClassifyNode initialized. Waiting for images...")

        while not rospy.is_shutdown():
            if self.latest_image is not None:
                self.process_image()
            self.rate.sleep()

    def image_callback(self, msg):
        self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def process_image(self):
        img = self.latest_image

        result = Bool()

        gap_prediction, vis_img, edge_img = self.detector(img)
        if self.save:
            cv2.imwrite(self.save_dir + f'/{self.frame_count:04}.png', img)
        result.data = gap_prediction

        self.result_pub.publish(result)
        self.frame_count += 1
        # vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="rgb8")
        self.vis_pub.publish(vis_msg)
        edge_msg = self.bridge.cv2_to_imgmsg(edge_img, encoding="rgb8")
        self.edge_pub.publish(edge_msg)

        rospy.loginfo(f"{robot_name} result: %s", result.data)
        

if __name__ == '__main__':
    robot_name = sys.argv[1] 
    assert(robot_name in ['yk_architect', 'yk_builder', 'yk_creator', 'yk_destroyer'])

    if robot_name == 'yk_destroyer':
        detector = gap_detector.GapDetector(robot_name)
    else:
        detector = gap_detector.GapDetector(robot_name)
        
    rospy.init_node(f'{robot_name}_gap_classify_node', anonymous=False)
    node = GapClassifyNode(robot_name, detector, save=True)
    rospy.spin()