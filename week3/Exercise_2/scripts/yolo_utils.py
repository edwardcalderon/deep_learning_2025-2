# yolo_utils.py
import os
import cv2
import numpy as np
from pathlib import Path
from IPython.display import display, Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm
import torch

class YOLOv8Detector:
    def __init__(self, model_size='yolov8n.pt'):
        """
        Initialize YOLOv8 detector with specified model size.
        Available sizes: 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 
                        'yolov8m.pt' (medium), 'yolov8l.pt' (large), 
                        'yolov8x.pt' (xlarge)
        """
        self.model = YOLO(model_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def detect_image(self, image_path, conf=0.5, save=False, output_dir='results'):
        """Detect objects in a single image"""
        results = self.model.predict(source=image_path, conf=conf)
        
        if save:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, results[0].plot())
            return output_path
        return results[0].plot()
    
    def detect_video(self, video_path, conf=0.5, output_path='output.mp4'):
        """Process video and save detection results"""
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame, conf=conf)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
        cap.release()
        out.release()
        return output_path

def download_kaggle_dataset(dataset_name, output_dir='dataset'):
    """Download dataset from Kaggle"""
    import zipfile
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    os.makedirs(output_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
    
    return os.path.join(output_dir, dataset_name.split('/')[-1])

def compare_models(image_path, model_sizes=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']):
    """Compare different YOLOv8 models on the same image"""
    results = {}
    for model_size in model_sizes:
        print(f"Processing with {model_size}...")
        detector = YOLOv8Detector(model_size)
        results[model_size] = detector.detect_image(image_path, conf=0.5)
    
    # Plot comparison
    plt.figure(figsize=(20, 10))
    for i, (model_name, img) in enumerate(results.items(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Model: {model_name}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return results

def create_yolo_dataset_structure(dataset_path, output_dir='yolo_dataset'):
    """
    Create YOLO dataset structure from raw dataset
    Expected structure:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    # Implementation depends on the dataset structure
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Create data.yaml
    data_yaml = f"""
    train: {os.path.join(output_dir, 'images/train')}
    val: {os.path.join(output_dir, 'images/val')}
    nc: 1  # Number of classes
    names: ['pothole']  # Class names
    """
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(data_yaml)
    
    return os.path.join(output_dir, 'data.yaml')