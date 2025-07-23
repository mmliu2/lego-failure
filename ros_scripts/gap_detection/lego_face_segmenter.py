from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer, ColorMode
import numpy as np


class LegoFaceSegmenter:
    def __init__(self, dataset_path="./data/lego-gap-3/train/", output_dir="./gap_detection/output", train=False):
        register_coco_instances("my_dataset", {}, dataset_path + "/_annotations.coco.json", dataset_path)

        print(DatasetCatalog.get("my_dataset"))
        print(MetadataCatalog.get("my_dataset").thing_classes)

        cfg = get_cfg()

        # Use a pretrained model from model zoo (e.g. Mask R-CNN)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        cfg.DATASETS.TRAIN = ("my_dataset",)
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

        self.ones_kernel = np.ones((3, 3), np.uint8)


    def get_top_bottom_masks(self, mask1, mask2):
        def center_y(mask):
            ys, xs = np.nonzero(mask)
            if len(ys) == 0:
                return float('inf')  # Empty mask
            return np.mean(ys)

        cy1 = center_y(mask1)
        cy2 = center_y(mask2)

        if cy1 < cy2:
            return mask1, mask2
        else:
            return mask2, mask1

    def best_fit_line(self, x, y):
        # Fit line y = mx + b using least squares
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, b
    
    def draw_mask_outline(self, img, mask, color):
        mask_img = mask.astype(np.uint8)
        edge_img = cv2.dilate(mask_img - cv2.erode(mask_img, self.ones_kernel), self.ones_kernel) # get inner edge
        img[edge_img!=0] = color
        

    def segment(self, img, rotation=-1):
        img = img.copy()
        outputs = self.predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("my_dataset_train"), scale=1.2, instance_mode=ColorMode.SEGMENTATION)
        segmented_img = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

        instances = outputs["instances"].to("cpu")
        
        # Get top 2 predictions
        sorted_scores = instances.scores.argsort(descending=True)
        sorted_instances = instances[sorted_scores]
        img = np.rot90(img, k=rotation)
        masks = np.rot90(sorted_instances.pred_masks.numpy(), k=rotation, axes=(1, 2)) 

        if len(masks) < 2: 
            print(f"LegoFaceSegmenter: less than 2 predictions ({len(masks)})")
            combined_mask = np.max(masks, axis=0)
            self.draw_mask_outline(img, combined_mask, color=(255, 0, 255))
            return segmented_img, img, []
        
        masks = masks[:2]      # (N, H, W) binary masks

        if np.any(masks[0] & masks[1]): 
            print(f"LegoFaceSegmenter: masks overlap")
            self.draw_mask_outline(img, masks[0], color=(0, 255, 0))
            self.draw_mask_outline(img, masks[1], color=(255, 0, 0))
            return segmented_img, img, []

        top_mask, bottom_mask = self.get_top_bottom_masks(masks[0], masks[1])
    
        lower_line_kernel = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.uint8)

        upper_line_kernel = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.uint8)
        
        lines = []

        for mask, kernel in ((top_mask, lower_line_kernel), (bottom_mask, upper_line_kernel)):
            # Kernel specifying which neighbors to check (horizontal 1D)
            mask_img = mask.astype(np.uint8)
            mask_img = cv2.dilate(cv2.erode(mask_img, self.ones_kernel), self.ones_kernel) # smoothing
            edge_img = mask_img - cv2.erode(mask_img, self.ones_kernel) # get inner edge

            # Keep only pixels where neighbor count == 0 (no white neighbors in specified pattern)
            neighbor_white_count = cv2.filter2D(mask_img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
            nbr_mask = (neighbor_white_count == 0)

            # Final output: keep edge pixels only where mask is true
            valid_points = edge_img * nbr_mask.astype(np.uint8)

            ys, xs = np.where(valid_points)
            xs = xs.astype(np.float32)
            ys = ys.astype(np.float32)
            m, b = self.best_fit_line(xs, ys)
            print(f"Fitted line: y = {m:.3f}x + {b:.3f}")
            lines.append((m, b))

        self.draw_mask_outline(img, top_mask, color=(255, 150, 0))
        self.draw_mask_outline(img, bottom_mask, color=(255, 0, 0))
        img = np.ascontiguousarray(img, dtype=np.uint8)
        for line in lines:
            m, b = line
            x1, x2 = 0, img.shape[1] - 1
            y1 = int(m * x1 + b)
            y2 = int(m * x2 + b)
            cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        return segmented_img, img, lines
    

if '__name__' == 'main':
    dataset_path = "./data/lego-gap-2/train/"
    segmenter = LegoFaceSegmenter(dataset_path=dataset_path, output_dir="./output", train=False)
    print('LegoFaceSegmenter initialized.')


    files = [dataset_path + f for f in os.listdir(dataset_path) if f[-4:] == '.jpg']

    for i in range(0, 3):
        img = cv2.imread(files[i])

        segmented_img, display_img, lines = segmenter.segment(img)

        segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
        plt.imshow(segmented_img)
        plt.show()
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        plt.imshow(display_img)
        plt.show()