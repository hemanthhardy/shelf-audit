import torch
import clip
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# Load the CLIP model and preprocessing pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)  # Use "ViT-L/14" for a larger model

# Preprocessing function (based on CLIP's requirements)
def preprocess_image(image):
    return preprocess(image).unsqueeze(0).to(device)

# Feature extraction function
def extract_features(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
    return image_features.cpu().numpy().squeeze()  # Return NumPy array


# Crop the image based on bounding box and extract features
def extract_features_from_bb(image, bb_boxes):
    features = []
    
    for bb in tqdm(bb_boxes):
        # Crop the image based on bounding box (bb: [x_min, y_min, x_max, y_max])
        cropped_image = image.crop((bb[0], bb[1], bb[2], bb[3]))
        
        # Extract features for each cropped region
        features.append(extract_features(cropped_image))
    
    return np.array(features)

# Perform DBSCAN clustering on the extracted features
def cluster_detections(image, bb_boxes):
    # Extract the product features
    print("feature extraction started")
    features = extract_features_from_bb(image, bb_boxes)
    
    # Apply DBSCAN clustering
    print("Clustering strated")
    dbscan = DBSCAN(eps=0.15, min_samples=1, metric="cosine").fit(features)
    print(dbscan.labels_)
    return dbscan.labels_.tolist()
