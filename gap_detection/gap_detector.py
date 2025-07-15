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
    def __init__(self, center, size, theta, 
                 detected_angle_tol=15, distance_tol=15,
                 display=False):
        
        self.size = int(size)

        self.detected_angle_min = (90+theta-detected_angle_tol)%180
        self.detected_angle_max = (90+theta+detected_angle_tol)%180
        self.distance_tol_left = distance_tol
        self.distance_tol_right = 10

        self.M, src_pts = self.get_rotated_square_transform(center, size, theta)
        self.M_inv = np.linalg.inv(self.M)
        self.src_pts = tuple(tuple(map(int, pt)) for pt in src_pts)

        self.hough_scale = 0.5

        self.brightness_threshold = 110

        self.display = display


    def get_rotated_square_transform(self, center, size, theta):
        theta = theta / 180 * np.pi # degrees to radians
        half = size / 2.0

        # Define destination square (upright)
        dst_pts = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype=np.float32)

        # Define source square (rotated around center)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        src_pts = np.array([
            [-half, -half],
            [ half, -half],
            [ half,  half],
            [-half,  half]
        ], dtype=np.float32)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_pts = (rotation @ src_pts.T).T + np.asarray(center)
        src_pts = rotated_pts.astype(np.float32)

        # Get perspective transform and warp
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        return M, src_pts

    def get_lines(self, cropped_img):
        hough_th = 50

        blurred = cv2.GaussianBlur(cropped_img, (3, 3), 0)
        edges_b = self.local_binarization(blurred[:, :, 0])
        edges_g = self.local_binarization(blurred[:, :, 1])
        edges_r = self.local_binarization(blurred[:, :, 2])
        edges_combined = np.minimum.reduce([edges_b, edges_g, edges_r])
        edges_combined = self.local_binarization(edges_combined)
        grayscale = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
        edges_combined[grayscale > self.brightness_threshold] = 255
        
        edges_combined[int(self.size*0.1):int(self.size*0.3), :] = 255
        edges_combined[int(self.size*0.7):int(self.size*0.9), :] = 255
        edge_img = 255 - cv2.resize(edges_combined, None, fx=self.hough_scale, fy=self.hough_scale, interpolation=cv2.INTER_AREA)
        edge_img[edge_img > 0] = 1

        if self.display: # show edges
            # self.imshow_no_axis(edges_r)
            # self.imshow_no_axis(edges_g)
            # self.imshow_no_axis(edges_b)
            edge_img_copy = edge_img.copy()
            cv2.line(edge_img_copy, (int(self.size//2*self.hough_scale), 0), 
                                (int(self.size//2*self.hough_scale), int(self.size*self.hough_scale)), 
                                (1, 0, 0), 1)
            cv2.line(edge_img_copy, (int((self.size//2-self.distance_tol_left)*self.hough_scale), 0), 
                                (int((self.size//2-self.distance_tol_left)*self.hough_scale), int(self.size*self.hough_scale)), 
                                (1, 0, 0), 1)
            cv2.line(edge_img_copy, (int((self.size//2+self.distance_tol_right)*self.hough_scale), 0), 
                                (int((self.size//2+self.distance_tol_right)*self.hough_scale), int(self.size*self.hough_scale)), 
                                (1, 0, 0), 1)
            self.imshow_no_axis(edge_img_copy)

        lines = cv2.HoughLinesP(edge_img, rho=1, theta=np.pi/180, threshold=hough_th,
                                minLineLength=self.size * self.hough_scale * 0.3, maxLineGap=self.size * self.hough_scale * 0.2)
        return lines

    def angle_between_points(self, line):
        x1, y1, x2, y2 = line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle % 180  # Normalize to [0, 180)
    
    def get_topmost_line(self, cropped_img, lines):
        # get topmost line
        top_line = None

        if lines is None:
            return None
        
        sorted_lines = sorted(lines[:, 0], key=lambda line: line[0] + line[2])

        top_line = None
        first_line_x, min_line_length = None, None
        # valid_lines = []
        for line in sorted_lines:
            angle = self.angle_between_points(line)
            if self.detected_angle_min <= angle <= self.detected_angle_max:
                # valid_lines.append([line])

                x1 = int(line[0]/self.hough_scale)
                y1 = int(line[1]/self.hough_scale)
                x2 = int(line[2]/self.hough_scale)
                y2 = int(line[3]/self.hough_scale)
                length = ((x2-x1)**2 + (y2-y1)**2)**0.5
                
                if first_line_x is None:
                    first_line_x, min_line_length = (x1 + x2)//2, length
                    top_line = (x1, y1, x2, y2)
                elif (x1 + x2)//2 <= first_line_x + 10 and length >= min_line_length*1.5:
                    min_line_length = length
                    top_line = (x1, y1, x2, y2)
                else:
                    break
        # lines = valid_lines
        # if valid_lines is None:
        #     print("No valid lines")
        #     return top_line
    
        if self.display: # show candidates
            # scaled down image
            lines_img = cv2.resize(cropped_img, None, fx=self.hough_scale, fy=self.hough_scale, interpolation=cv2.INTER_AREA)
            cv2.line(lines_img, (int(self.size//2*self.hough_scale), 0), 
                                (int(self.size//2*self.hough_scale), int(self.size*self.hough_scale)), 
                                (255, 0, 0), 1)
            cv2.line(lines_img, (int((self.size//2-self.distance_tol_left)*self.hough_scale), 0), 
                                (int((self.size//2-self.distance_tol_left)*self.hough_scale), int(self.size*self.hough_scale)), 
                                (255, 0, 0), 1)
            cv2.line(lines_img, (int((self.size//2+self.distance_tol_right)*self.hough_scale), 0), 
                                (int((self.size//2+self.distance_tol_right)*self.hough_scale), int(self.size*self.hough_scale)), 
                                (255, 0, 0), 1)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            self.imshow_no_axis(lines_img)

        return top_line
    
    def is_valid_line(self, top_line):
        expected_x = self.size / 2
        is_valid_dist = -self.distance_tol_right <= expected_x - top_line[0] <= self.distance_tol_left and \
                        -self.distance_tol_right <= expected_x - top_line[2] <= self.distance_tol_left
        
        if not is_valid_dist:
            # print(f'invalid x values: {top_line[0]}, {top_line[2]} (expected: {expected_x-self.distance_tol_left, expected_x+self.distance_tol_right})')
            return False
        else:
            return True

    def extend_line_to_square_edges(self, line):
        x1, y1, x2, y2 = line

        if y1 == y2:
            return (0, y1, self.size, y1)
        if x1 == x2:
            return (x1, 0, x1, self.size)

        # Line equation: y = m * x + b
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Solve for x at y = 0 and y = size
        x_at_y0 = (0 - b) / m
        x_at_ysize = (self.size - b) / m

        return (x_at_y0, 0, x_at_ysize, self.size)

    def transform_line_to_og(self, line): # xmin, ymin, xmax, ymax
        line_pts_h = np.asarray([(line[0], line[1], 1), (line[2], line[3], 1)])
        transformed_pts_h = self.M_inv @ line_pts_h.T
        transformed_pts = (transformed_pts_h[:2] / transformed_pts_h[2]).T
        transformed_line_pt1 = (int(transformed_pts[0,0]), int(transformed_pts[0,1]))
        transformed_line_pt2 = (int(transformed_pts[1,0]), int(transformed_pts[1,1]))
        return transformed_line_pt1, transformed_line_pt2
    

    def local_binarization(self, img):
        """
        Apply local (adaptive) binarization with a 5x5 kernel.
        
        Parameters:
            img: Grayscale image as a 2D NumPy array (uint8).
            
        Returns:
            Binary image as a 2D NumPy array (uint8, values 0 or 255).
        """
        binary = cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            thresholdType=cv2.THRESH_BINARY,
            blockSize=9,
            C=2  # small constant subtracted from mean; tune as needed
        )
        binary = cv2.medianBlur(binary, ksize=3)  # or 5, must be odd
        return binary
    

    def __call__(self, img_np):
        # torch.manual_seed(0)
        # np.random.seed(0)

        # img_np = np.array(img)
        cropped_img = cv2.warpPerspective(img_np, self.M, (self.size, self.size))

        lines = self.get_lines(cropped_img)
        top_line = self.get_topmost_line(cropped_img, lines)
        
        if top_line is None:
            if self.display:
                print('No gap detected (no lines found):')
                self.visualize(img_np, top_line)
            return False, self.visualize(img_np, top_line, result=False, display=False)
        
        top_line = self.extend_line_to_square_edges(top_line)

        if not self.is_valid_line(top_line):
            if self.display:
                print('Gap detected:')
                self.visualize(img_np, top_line)
            return True, self.visualize(img_np, top_line, result=True, display=False)
        else:
            if self.display:
                print('No gap detected:')
                self.visualize(img_np, top_line)
            return False, self.visualize(img_np, top_line, result=False, display=False)


    def visualize(self, img, top_line, result=None, display=True):
        img = img.copy()

        def draw_line(pt1, pt2, color=(255, 0, 0)):
            cv2.line(img, pt1, pt2, color=color, thickness=1)

        def midpoint(pt1, pt2, midpoint=0.5):
            return (int(pt1[0]*midpoint+pt2[0]*(1-midpoint)),
                    int(pt1[1]*midpoint+pt2[1]*(1-midpoint)))
        
        topleft, botleft, botright, topright = self.src_pts

        draw_line(topleft, topright)
        draw_line(topright, botright)
        draw_line(botright, botleft)
        draw_line(botleft, topleft)

        draw_line(midpoint(topleft, botleft), midpoint(topright, botright))
        draw_line(midpoint(topleft, topright), midpoint(botleft, botright))

        draw_line(midpoint(midpoint(topleft, topright, 0.25), midpoint(botleft, botright, 0.25), 0.55), 
                  midpoint(midpoint(topleft, topright, 0.25), midpoint(botleft, botright, 0.25)))
        draw_line(midpoint(midpoint(topleft, topright, 0.75), midpoint(botleft, botright, 0.75), 0.55), 
                  midpoint(midpoint(topleft, topright, 0.75), midpoint(botleft, botright, 0.75)))
        
        tol_line_left = (int((self.size//2-self.distance_tol_left)), 0, 
                    int((self.size//2-self.distance_tol_left)), int(self.size))
        tol_line_pt1, tol_line_pt2 = self.transform_line_to_og(tol_line_left)
        draw_line(tol_line_pt1, tol_line_pt2, color=(0, 0, 0))
        tol_line_right = (int((self.size//2+self.distance_tol_right)), 0, 
                    int((self.size//2+self.distance_tol_right)), int(self.size))
        tol_line_pt1, tol_line_pt2 = self.transform_line_to_og(tol_line_right)
        draw_line(tol_line_pt1, tol_line_pt2, color=(0, 0, 0))
        
        if top_line is not None:
            top_line_pt1, top_line_pt2 = self.transform_line_to_og(top_line)
            draw_line(top_line_pt1, top_line_pt2, color=(0, 255, 0))

        if result is not None:
            text = f"gap: {result}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 2)
            cv2.rectangle(img, (0, 0), (263 + 20, text_h + 20), (0, 0, 0), -1)
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
