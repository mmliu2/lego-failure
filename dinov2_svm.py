import numpy as np
import torch
from PIL import Image
import os
import json
from tqdm.notebook import tqdm

import torchvision.transforms as T

from sklearn import svm
import pickle


def load_image(img: str, transformation, augmentation=None) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)

    transformed_img = transformation(img)
    if augmentation:
        transformed_img = augmentation(transformed_img)

    return transformed_img[:3].unsqueeze(0)

def compute_embeddings(files: list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}
    
    with torch.no_grad():
        for file in tqdm(files):
            embeddings = dinov2_vits14(load_image(file, transformation).to(device))
            all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()
            
    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings


if __name__ == "__main__":
    train_dir = './data/train_images_061225'
    model_save_path = './models/pick_place_svm_061325.pkl'

    labels = {}
    for folder in os.listdir(train_dir):
        for file in os.listdir(os.path.join(train_dir, folder)):
            if file.endswith(".png"):
                full_name = os.path.join(train_dir, folder, file)
                labels[full_name] = folder
    files = labels.keys()

    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)

    transformation = T.Compose([T.ToTensor(), 
                                T.Resize(224), 
                                T.CenterCrop(224), 
                                T.Normalize([0.5], [0.5]),
                                T.Grayscale(num_output_channels=3),
                                ])

    embeddings = compute_embeddings(files)
    clf = svm.SVC(gamma='scale')
    y = [labels[file] for file in files]
    print(len(embeddings.values()))

    embedding_list = list(embeddings.values())
    clf.fit(np.array(embedding_list).reshape(-1, 384), y)

    with open(model_save_path, 'wb') as file:
        pickle.dump(clf, file)
    
    print("Done.")