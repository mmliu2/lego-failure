import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm.notebook import tqdm

import os
import pickle
import warnings
import matplotlib.pyplot as plt

def load_image(img: str, transformation, augmentation=None) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)

    transformed_img = transformation(img)
    if augmentation:
        transformed_img = augmentation(transformed_img)

    return transformed_img[:3].unsqueeze(0)

def results(paths, expected_output=''):
    count = 0
    for input_file in tqdm(paths, total=len(paths)):
        new_image = load_image(input_file, transformation)

        with torch.no_grad():
            embedding = dinov2_vits14(new_image.to(device))

            prediction = clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))

            # print("Predicted class: " + prediction[0])
            if prediction[0] == expected_output:
                count += 1
            else:
                print(input_file)
                # plt.imshow(Image.open(input_file))
                # plt.show()
    print(count/len(paths))

if __name__ == "__main__":
    model_path = os.getcwd() + './pick_place_svm_061225.pkl'
    test_path = os.getcwd() + '../data/test_images/'

    warnings.filterwarnings('ignore', category=DeprecationWarning)


    # load model
    with open(model_path, 'rb') as file:
        clf = pickle.load(file)

    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)

    # image paths
    success_test_dir = os.path.join(test_path, 'new_success/')
    success_paths = ([os.path.join(success_test_dir, f) for f in os.listdir(success_test_dir)])
 
    failure_test_dir = os.path.join(test_path, 'failure/')
    failure_paths = ([os.path.join(failure_test_dir, f) for f in os.listdir(failure_test_dir)])

    transformation = T.Compose([T.ToTensor(), 
                              T.Resize(224), 
                              T.CenterCrop(224), 
                              T.Normalize([0.5], [0.5]),
                              T.Grayscale(num_output_channels=3),
                              ])

    results(success_paths, 'success')
    results(failure_paths, 'failure')

