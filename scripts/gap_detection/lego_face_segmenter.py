from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer, ColorMode
import numpy as np
import argparse
from sklearn.linear_model import RANSACRegressor, LinearRegression

class LegoFaceSegmenter:
    def __init__(self, dataset_path="./data/lego-gap-6/train/", output_dir="./gap_detection/output", train=False):
        self.width, self.height = 640, 480

        register_coco_instances("my_dataset", {}, dataset_path + "/_annotations.coco.json", dataset_path)
        print(DatasetCatalog.get("my_dataset"))
        print(MetadataCatalog.get("my_dataset").thing_classes)

        cfg = get_cfg()

        # Use a pretrained model from model zoo (e.g. Mask R-CNN)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.DATASETS.TRAIN = ("my_dataset",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # Adjust as needed
        cfg.SOLVER.MAX_ITER = 3000    # Adjust based on dataset size
        cfg.SOLVER.STEPS = []         # No LR decay
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust to number of classes in your dataset
        cfg.OUTPUT_DIR = output_dir
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        trainer = DefaultTrainer(cfg)

        if train:
            trainer.resume_or_load(resume=False)
            trainer.train()
        else:
            trainer.resume_or_load(resume=True)

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        self.predictor = DefaultPredictor(cfg)
        
        ###############################################

        self.ones_kernel = np.ones((3, 3), np.uint8)
        self.lower_line_kernel = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.uint8)
        self.upper_line_kernel = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        # visualization
        self.fail_fill = (150, 150, 150)
        self.fail_line = (0, 0, 0)
        self.gap_fill = (150, 150, 150)
        self.gap_line = (0, 0, 255)
        self.nogap_fill = (150, 150, 150)
        self.nogap_line = (0, 255, 0)

        # thresholds for gap detection
        self.mask_overlap_tol = 0.02*self.width*self.height # area        
        self.line_max_slope = 1.0*self.height # endpoint y difference  
        self.line_overlap_tol = 0.1*self.height
        self.line_gap_tol = 0.08*self.height   
        self.line_slope_diff_tol = 0.08*self.height
        

    def segment(self, img, rotation=-1): # expect bgr image
        img = cv2.resize(img, (self.width, self.height))
        img = img[40:460, 40:460] # crop out tool parts

        outputs = self.predictor(img)

        instances = outputs["instances"].to("cpu")
        sorted_idxs = instances.scores.argsort(descending=True)
        high_score_mask = instances.scores >= 0.9 ###
        filtered_indexes = sorted_idxs[high_score_mask[sorted_idxs]]
        sorted_instances = instances[filtered_indexes]
        
        # orient image and masks
        vis_img = np.rot90(img, k=rotation) 
        vis_img = np.ascontiguousarray(vis_img, dtype=np.uint8)
        masks = np.rot90(sorted_instances.pred_masks.numpy(), k=rotation, axes=(1, 2))
        masks = np.ascontiguousarray(masks, dtype=np.uint8)

        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("my_dataset_train"), scale=1.2, instance_mode=ColorMode.SEGMENTATION)
        seg_img = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image() # for display
        return masks, vis_img, seg_img

    def get_mask_pair(self, masks, edge_extension=0.1, min_contact=0.2):
        # top mask must contact roughly at least 20% of top edge and 0% of bottom edge of image
        top_mask = None
        bottom_mask = None

        for mask in masks[::-1]:
            convex_mask = self.convexify_and_fill(mask)
            edge_y = int(self.height*edge_extension)
            top_edge_contact = np.sum(convex_mask[:edge_y])/(edge_y*self.width)
            bottom_edge_contact = np.sum(convex_mask[-edge_y:])/(edge_y*self.width)
            if top_edge_contact > min_contact and bottom_edge_contact == 0:
                top_mask = convex_mask
            elif bottom_edge_contact > min_contact and top_edge_contact == 0:
                bottom_mask = convex_mask

        return top_mask, bottom_mask
        
    def convexify_and_fill(self, mask):
        mask_img = mask.astype(np.uint8)

        # Find external contours
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask_img)

        cnt = max(contours, key=cv2.contourArea) # Find largest contour (or loop through all)
        hull = cv2.convexHull(cnt) # Get convex hull of contour
        convex_mask = np.zeros_like(mask_img) # Draw filled convex hull into a new mask
        cv2.drawContours(convex_mask, [hull], -1, color=1, thickness=-1)

        # Fill small holes (optional)
        convex_mask = cv2.morphologyEx(convex_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        return convex_mask

    def best_fit_line(self, mask_img, kernel): 
        edge_img = mask_img - cv2.erode(mask_img, self.ones_kernel) # get inner edge
        neighbor_white_count = cv2.filter2D(mask_img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        nbr_mask = (neighbor_white_count == 0)
        valid_points = edge_img * nbr_mask.astype(np.uint8)

        y, x = np.where(valid_points)
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Prepare data
        X = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # Fit using RANSAC
        ransac = RANSACRegressor(LinearRegression(), residual_threshold=2.0)
        ransac.fit(X, y)

        # Get model params
        slope = ransac.estimator_.coef_[0][0]
        intercept = ransac.estimator_.intercept_[0]

        return slope, intercept
    
    def draw_mask_outline(self, img, mask, color=(255, 255, 255)):
        if mask is None: return
        mask_img = mask.astype(np.uint8)
        img[mask_img!=0] = (color + img[mask_img!=0]) // 2
        
        # outline
        edge_img = mask_img - cv2.erode(cv2.erode(mask_img, self.ones_kernel, borderValue=0), self.ones_kernel, borderValue=0) # get inner edge
        img[edge_img!=0] = (255, 255, 255)
        

    def classify_gap(self, img):
        masks, vis_img, seg_img = self.segment(img)

        if len(masks) < 2: 
            print(f"LegoFaceSegmenter: less than 2 predictions ({len(masks)})")
            if len(masks) == 1:
                self.draw_mask_outline(vis_img, masks[0], color=self.fail_fill)
            return -1, seg_img, vis_img
        
        top_mask, bottom_mask = self.get_mask_pair(masks) # (N, H, W) binary masks

        if top_mask is None or bottom_mask is None:
            print(f"LegoFaceSegmenter: top or bottom mask could not be found")
            return -1, seg_img, vis_img
        elif np.sum(top_mask & bottom_mask) > self.mask_overlap_tol: 
            print(f"LegoFaceSegmenter: masks overlap significantly")
            self.draw_mask_outline(vis_img, top_mask, color=self.fail_fill)
            self.draw_mask_outline(vis_img, bottom_mask, color=self.fail_fill)
            return -1, seg_img, vis_img
       
        top_slope, top_intercept = self.best_fit_line(top_mask, self.lower_line_kernel)
        bottom_slope, bottom_intercept = self.best_fit_line(bottom_mask, self.upper_line_kernel)

        a0, a1 = int(top_intercept), int(self.width*top_slope + top_intercept)
        b0, b1 = int(bottom_intercept), int(self.width*bottom_slope + bottom_intercept)

        if abs(a1-a0) > self.line_max_slope or abs(b1-b0) > self.line_max_slope: 
            print(f"LegoFaceSegmenter: line slope too steep")
            result = -1
            fill_color, line_color = self.fail_fill, self.fail_line
        elif a0-b0 > self.line_overlap_tol or a1-b1 > self.line_overlap_tol: 
            print(f"LegoFaceSegmenter: lines overlap significantly")
            result = -1 
            fill_color, line_color = self.fail_fill, self.fail_line
        elif abs((b0-a0) - (b1-a1)) > self.line_slope_diff_tol: 
            result = 1 
            fill_color, line_color = self.gap_fill, self.gap_line
        elif b0-a0 > self.line_gap_tol or b1-a1 > self.line_gap_tol: 
            result = 1 # gap
            fill_color, line_color = self.gap_fill, self.gap_line
        else: 
            result = 0
            fill_color, line_color = self.nogap_fill, self.nogap_line

        self.draw_mask_outline(vis_img, top_mask, color=fill_color)
        self.draw_mask_outline(vis_img, bottom_mask, color=fill_color)
        cv2.line(vis_img, (0, a0), (self.width, a1), color=line_color, thickness=4)
        cv2.line(vis_img, (0, b0), (self.width, b1), color=line_color, thickness=4)
        
        return result, seg_img, vis_img
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--dataset', default='lego-gap-6')
    args = parser.parse_args()

    DATASET_NAME = args.dataset
    dataset_path = f"/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/{DATASET_NAME}/train/"
    output_path = f"/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/models/{DATASET_NAME}_output"

    segmenter = LegoFaceSegmenter(dataset_path=dataset_path, output_dir=output_path, train=args.train)
    print('LegoFaceSegmenter initialized.')

    files = [dataset_path + f for f in os.listdir(dataset_path) if f[-4:] == '.jpg']

    for i in range(0, 3):
        img = cv2.imread(files[i])
        result, seg_img, vis_img = segmenter.classify_gap(img)
        cv2.imwrite(f'/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/lego_segmenter_examples/{i:000}_seg.png', seg_img)
        cv2.imwrite(f'/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/lego_segmenter_examples/{i:000}_vis.png', vis_img)