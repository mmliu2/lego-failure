import os
import json
from roboflow import Roboflow

################### 
# DATA_PATH = '/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/'

rf = Roboflow(api_key="17jLrqsgdEqy4XItF5kA")
project = rf.workspace("lego-gombf").project("lego-pickplace-01gil")
version = project.version(5)
dataset = version.download("folder")
                
                
# DATASET_NAME = 'pickplace_train' # edit
###################

# # Read JSON as string
# json_path = os.path.join(DATA_PATH, DATASET_NAME, "train/_annotations.coco.json")
# with open(json_path, "r", encoding="utf-8") as f:
#     json_str = f.read()

# json_str = json_str.replace('"categories":[{"id":0,"name":"lego-face","supercategory":"none"},{"id":1,"name":"lego-face","supercategory":"lego-face"}]', 
#                             '"categories":[{"id":1,"name":"lego-face","supercategory":"lego-face"}]')

# # Write back to file
# with open(json_path, "w", encoding="utf-8") as f:
#     f.write(json_str)