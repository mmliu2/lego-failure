import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm

import os
import pickle
import warnings
import matplotlib.pyplot as plt

import math
import cv2


class GapDetector:
    def __init__(self, robot_name, detected_angle_tol=25, distance_tol=10,
                 display=False):
        
        # july 2025
        if robot_name == 'yk_destroyer':
            center_xy = (255, 295)
            unit_length = 160
            theta = 90 # TODO: adjust view pose so that theta ~= 0
        elif robot_name == 'yk_architect':
            center_xy = (305, 215)
            unit_length = 180
            theta = 91
        else:
            raise Exception('Invalid robot name')
        
        self.w = int(unit_length)
        self.h = self.w*2

        self.detected_angle_min = (-detected_angle_tol+90)%180-90
        self.detected_angle_max = (+detected_angle_tol+90)%180-90
        self.distance_tol_left = distance_tol
        self.distance_tol_right = 10

        self.M, src_pts = self.get_transform(center_xy, (self.h, self.w), theta)
        self.M_inv = np.linalg.inv(self.M)
        self.src_pts = tuple(tuple(map(int, pt)) for pt in src_pts)

        self.hough_scale = 1

        self.brightness_threshold = 160

        self.display = display


    def get_transform(self, center, size, theta):
        theta = theta / 180 * np.pi # degrees to radians
        height, width = size
        half_h = height / 2.0
        half_w = width / 2.0

        # Define destination square (upright)
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # Define source square (rotated around center)
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)

        # xy pairs
        src_pts = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h]
        ], dtype=np.float32)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_pts = (rotation @ src_pts.T).T + np.asarray(center)
        src_pts = rotated_pts.astype(np.float32)

        # Get perspective transform and warp
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        return M, src_pts
    

    def extract_edges(self, cropped_img):
        blurred = cv2.GaussianBlur(cropped_img, (5, 5), 0)
        edges_b = cv2.Canny(blurred[:, :, 0], 30, 70)
        edges_g = cv2.Canny(blurred[:, :, 1], 30, 70)
        edges_r = cv2.Canny(blurred[:, :, 2], 30, 70)
        edges_combined = np.minimum.reduce([edges_b, edges_g, edges_r])
        edges_combined = self.local_binarization(edges_combined)
        grayscale = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
        # self.imshow_no_axis(grayscale > self.brightness_threshold)
        edges_combined[grayscale > self.brightness_threshold] = 255
        
        # edges_combined[int(self.width*0.1):int(self.width*0.3), :] = 255
        # edges_combined[int(self.width*0.7):int(self.width*0.9), :] = 255
        # edges_combined = cv2.dilate(255-edges_combined, np.ones((3, 3), np.uint8), dst=None, anchor=(-1, -1), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=None)
        
        edge_img = 255 - cv2.resize(edges_combined, None, fx=self.hough_scale, fy=self.hough_scale, interpolation=cv2.INTER_AREA)
        edge_img[edge_img > 0] = 1
        # edge_img = cv2.dilate(edge_img, np.ones((3, 3), np.uint8), dst=None, anchor=(-1, -1), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=None)
    
        return edge_img
    
    # def extract_edges(self, cropped_img):
    #     blurred = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    #     edges_b = self.local_binarization(blurred[:, :, 0])
    #     edges_g = self.local_binarization(blurred[:, :, 1])
    #     edges_r = self.local_binarization(blurred[:, :, 2])
    #     edges_combined = np.minimum.reduce([edges_b, edges_g, edges_r])
    #     edges_combined = self.local_binarization(edges_combined)
    #     grayscale = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    #     # self.imshow_no_axis(grayscale > self.brightness_threshold)
    #     edges_combined[grayscale > self.brightness_threshold] = 255
        
    #     # edges_combined[int(self.width*0.1):int(self.width*0.3), :] = 255
    #     # edges_combined[int(self.width*0.7):int(self.width*0.9), :] = 255
    #     # edges_combined = cv2.dilate(255-edges_combined, np.ones((3, 3), np.uint8), dst=None, anchor=(-1, -1), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=None)
        
    #     edge_img = 255 - cv2.resize(edges_combined, None, fx=self.hough_scale, fy=self.hough_scale, interpolation=cv2.INTER_AREA)
    #     edge_img[edge_img > 0] = 1
    #     # edge_img = cv2.dilate(edge_img, np.ones((3, 3), np.uint8), dst=None, anchor=(-1, -1), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=None)
    
    #     return edge_img

    def get_lines(self, edge_img, hough_th=100, hough_min_line_length=0.8, hough_max_line_gap=0.3):

        if self.display: # show edges
            # self.imshow_no_axis(edges_r)
            # self.imshow_no_axis(edges_g)
            # self.imshow_no_axis(edges_b)
            edge_img_copy = edge_img.copy()

            # tolerance lines
            for x_shift in (-self.distance_tol_left, 0, self.distance_tol_right):
                cv2.line(edge_img_copy, (0, int(self.h//2*self.hough_scale+x_shift)), 
                                (int(self.w*self.hough_scale), int(self.h//2*self.hough_scale+x_shift)), 
                                (1, 0, 0), 1)
            self.imshow_no_axis(edge_img_copy)

        lines = cv2.HoughLinesP(edge_img[:int(self.h*0.75)], rho=1, theta=np.pi/180, threshold=hough_th,
                                minLineLength=self.w * self.hough_scale * hough_min_line_length, 
                                maxLineGap=self.w * self.hough_scale * hough_max_line_gap)
        
        if lines is None:
            return None
        
        return [line for line in lines[:, 0] if \
                self.detected_angle_min <= 90+self.angle_between_points(line)%180-90 <= self.detected_angle_max]

    def angle_between_points(self, line):
        x1, y1, x2, y2 = line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle % 180  # Normalize to [0, 180)
    
    def get_topmost_line(self, img, lines):
        lines_img = cv2.resize(img, None, fx=self.hough_scale, fy=self.hough_scale, interpolation=cv2.INTER_AREA)

        # get topmost line
        top_line = None

        if lines is None:
            return None, lines_img
        
        sorted_lines = sorted(lines, key=lambda line: line[1] + line[3])

        top_line = None
        first_line_y, min_line_length = None, None
        # valid_lines = []
        for line in sorted_lines:
            # valid_lines.append([line])

            x1 = int(line[0]/self.hough_scale)
            y1 = int(line[1]/self.hough_scale)
            x2 = int(line[2]/self.hough_scale)
            y2 = int(line[3]/self.hough_scale)
            length = ((x2-x1)**2 + (y2-y1)**2)**0.5
            
            if first_line_y is None:
                first_line_y, min_line_length = (y1 + y2)//2, length
                top_line = (x1, y1, x2, y2)
            elif (x1 + x2)//2 <= first_line_y + 10 and length >= min_line_length*1.5:
                min_line_length = length
                top_line = (x1, y1, x2, y2)
            else:
                break
        # lines = valid_lines
        # if valid_lines is None:
        #     print("No valid lines")
        #     return top_line
    
        # if self.display: # show candidates
            # scaled down image
        # lines_img = cv2.resize(img, None, fx=self.hough_scale, fy=self.hough_scale, interpolation=cv2.INTER_AREA)

        # tolerance lines
        for x_shift in (-self.distance_tol_left, 0, self.distance_tol_right):
            cv2.line(lines_img, (0, int(self.h//2*self.hough_scale+x_shift)), 
                            (int(self.w*self.hough_scale), int(self.h//2*self.hough_scale+x_shift)), 
                            (1, 0, 0), 1)
        for line in lines:
            x1, y1, x2, y2 = line #[0]
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # self.imshow_no_axis(lines_img)

        return top_line, lines_img
    
    def is_valid_line(self, top_line):
        expected_y = self.h / 2
        is_valid_dist = -self.distance_tol_right <= expected_y - top_line[1] <= self.distance_tol_left and \
                        -self.distance_tol_right <= expected_y - top_line[3] <= self.distance_tol_left
        
        if not is_valid_dist:
            return False
        else:
            return True

    def extend_line_to_square_edges(self, line):
        x1, y1, x2, y2 = line

        if y1 == y2:
            return (0, y1, self.w, y1)
        if x1 == x2:
            return (x1, 0, x1, self.h)

        # Line equation: y = m * x + b
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        y_at_x0 = b
        y_at_xsize = m * self.w + b

        return (0, y_at_x0, self.w, y_at_xsize)


    def local_binarization(self, img):
        binary = cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,
            C=3  # small constant subtracted from mean; tune as needed
        )
        binary = cv2.medianBlur(binary, ksize=3)  # or 5, must be odd
        return binary
    

    def __call__(self, img_np):
        # torch.manual_seed(0)
        # np.random.seed(0)

        # img_np = np.array(img)
        cropped_img = cv2.warpPerspective(img_np, self.M, (self.w, self.h))

        edge_img = self.extract_edges(cropped_img)
        lines = self.get_lines(edge_img, hough_th=70, hough_min_line_length=0.75, hough_max_line_gap=0.2)
        edge_img = 255*cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB) # for display
        # if lines is None:
        #     lines = self.get_lines(edge_img, hough_th=60, hough_min_line_length=0.8, hough_max_line_gap=0.2)
        
        top_line, lines_img = self.get_topmost_line(edge_img, lines)
        
        if top_line is None:
            result = 0
            if self.display:
                print('No gap detected (no lines found):')
                self.visualize(img_np, top_line, result)
        else:
            top_line = self.extend_line_to_square_edges(top_line)

            if not self.is_valid_line(top_line):
                result = 1
                if self.display:
                    print('Gap detected:')
                    self.visualize(img_np, top_line, result)
            else:
                result = 0
                if self.display:
                    print('No gap detected:')
                    self.visualize(img_np, top_line, result)
        
        # return result, self.visualize(img_np, top_line, result, display=False), 255*cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
        return result, self.visualize(img_np, top_line, result, display=False), lines_img #255*cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)


    def transform_line_to_og(self, line): # xmin, ymin, xmax, ymax
        line_pts_h = np.asarray([(line[0], line[1], 1), (line[2], line[3], 1)])
        transformed_pts_h = self.M_inv @ line_pts_h.T
        transformed_pts = (transformed_pts_h[:2] / transformed_pts_h[2]).T
        transformed_line_pt1 = (int(transformed_pts[0,0]), int(transformed_pts[0,1]))
        transformed_line_pt2 = (int(transformed_pts[1,0]), int(transformed_pts[1,1]))
        return transformed_line_pt1, transformed_line_pt2
    
    def visualize(self, img, top_line, result, display=True):
        def draw_line(pt1, pt2, color=(0, 0, 255), thickness=2):
            cv2.line(img, pt1, pt2, color=color, thickness=thickness)

        def midpoint(pt1, pt2, x_shift=0, y_shift=0, midpoint=0.5):
            return (int(pt1[0]*midpoint+pt2[0]*(1-midpoint) + x_shift),
                    int(pt1[1]*midpoint+pt2[1]*(1-midpoint) + y_shift))

        img = img.copy()
        
        topl, topr, botr, botl = self.src_pts

        draw_line(topl, topr)
        draw_line(botr, botl)
        draw_line(botl, topl)
        draw_line(topr, botr)

        draw_line(midpoint(topr, botr), midpoint(topl, botl))
        
        # studs
        for x_center_shift in (-self.w, 0, self.w):
            x_center, y_center = self.w//2 + x_center_shift, self.h//2
            stud_w = 0.32
            stud_h = 0.1
            stud_line1 = (x_center+self.w*stud_w, y_center-self.h*stud_h, x_center+self.w*stud_w, y_center)
            stud_line2 = (x_center-self.w*stud_w, y_center-self.h*stud_h, x_center-self.w*stud_w, y_center)
            stud_line3 = (stud_line1[0], stud_line1[1], stud_line2[0], stud_line2[1])
            for stud_line in (stud_line1, stud_line2, stud_line3):
                line_pt1, line_pt2 = self.transform_line_to_og(stud_line)
                draw_line(line_pt1, line_pt2)

        # tolerance lines
        for y in (-self.distance_tol_left, self.distance_tol_right):
            tol_line_y = int((self.h//2+y))
            tol_line = (0, tol_line_y, int(self.w), tol_line_y)
            line_pt1, line_pt2 = self.transform_line_to_og(tol_line)
            draw_line(line_pt1, line_pt2, color=(0, 0, 255))
            draw_line(line_pt1, line_pt2, color=(255, 255, 255))
        
        if top_line is not None:
            top_line_pt1, top_line_pt2 = self.transform_line_to_og(top_line)
            draw_line(top_line_pt1, top_line_pt2, color=(255, 0, 0) if result else (0, 255, 0), thickness=2)

        text = f"gap: {result}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 2)
        cv2.rectangle(img, (0, 0), (text_w + 10, text_h + 20), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 10 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)

        if display:
            plt.imshow(img)
            plt.axis('off')
            plt.show()

        return img


    def imshow_no_axis(self, img):
        plt.imshow(img)
        plt.axis('off')
        plt.show()
