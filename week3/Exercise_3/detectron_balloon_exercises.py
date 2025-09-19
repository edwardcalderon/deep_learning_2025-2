"""
Detectron2 Balloon Dataset Exercises

This script implements various exercises for object detection using Detectron2 on the balloon dataset.
The exercises include:
1. Reducing the number of training examples by half and comparing training time, predictions, and loss.
2. Increasing the learning rate and analyzing the results.
3. Modifying the test threshold for better prediction.
4. Increasing the maximum number of iterations and noting the impact.
5. Testing with a custom image and summarizing observations.
"""

import os
import json
import random
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Set up logger
setup_logger()

# Constants
DATASET_DIR = "balloon"
TRAIN_JSON = os.path.join(DATASET_DIR, "train/via_region_data.json")
VAL_JSON = os.path.join(DATASET_DIR, "val/via_region_data.json")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "val")
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Register the balloon dataset
def register_balloon_dataset(json_file, image_dir, dataset_name):
    register_coco_instances(dataset_name, {}, json_file, image_dir)
    return dataset_name

# Load and prepare the dataset
def load_and_prepare_dataset():
    # Register training and validation datasets
    train_dataset = register_balloon_dataset(TRAIN_JSON, TRAIN_IMG_DIR, "balloon_train")
    val_dataset = register_balloon_dataset(VAL_JSON, VAL_IMG_DIR, "balloon_val")
    
    # Get dataset metadata
    balloon_metadata = MetadataCatalog.get("balloon_train")
    dataset_dicts = DatasetCatalog.get("balloon_train")
    
    return balloon_metadata, dataset_dicts

# Function to reduce training dataset by half
def reduce_dataset(dataset_dicts, fraction=0.5):
    random.shuffle(dataset_dicts)
    reduced_size = int(len(dataset_dicts) * fraction)
    return dataset_dicts[:reduced_size]

# Function to train and evaluate model
def train_and_evaluate(cfg, output_dir, reduced_dataset=None):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create trainer
    trainer = DefaultTrainer(cfg)
    
    # If using reduced dataset, modify the data loader
    if reduced_dataset is not None:
        from detectron2.data import DatasetMapper, build_detection_train_loader
        
        class ReducedDatasetTrainer(DefaultTrainer):
            @classmethod
            def build_train_loader(cls, cfg):
                mapper = DatasetMapper(cfg, is_train=True)
                return build_detection_train_loader(cfg, dataset=reduced_dataset, mapper=mapper)
        
        trainer = ReducedDatasetTrainer(cfg)
    
    # Train the model
    start_time = time.time()
    trainer.resume_or_load(resume=False)
    trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate the model
    evaluator = COCOEvaluator("balloon_val", output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "balloon_val")
    metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    return trainer, metrics, training_time

# Function to visualize predictions
def visualize_predictions(cfg, dataset_dicts, metadata, output_dir, num_vis=3):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set threshold
    predictor = DefaultPredictor(cfg)
    
    for i, d in enumerate(random.sample(dataset_dicts, num_vis)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                      metadata=metadata,
                      scale=0.8,
                      instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Save visualization
        output_path = os.path.join(output_dir, f"prediction_{i}.jpg")
        cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
        print(f"Saved prediction to {output_path}")

def main():
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    balloon_metadata, dataset_dicts = load_and_prepare_dataset()
    
    # Exercise 1: Reduce training dataset by half
    print("\n--- Exercise 1: Reduce training dataset by half ---")
    reduced_dataset = reduce_dataset(dataset_dicts, fraction=0.5)
    print(f"Original dataset size: {len(dataset_dicts)}")
    print(f"Reduced dataset size: {len(reduced_dataset)}")
    
    # Base configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ("balloon_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # Base learning rate
    cfg.SOLVER.MAX_ITER = 1000  # Base number of iterations
    cfg.SOLVER.STEPS = []  # Do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Faster R-CNN parameter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only has one class (balloon)
    
    # Train with full dataset
    print("\nTraining with full dataset...")
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, "full_dataset")
    trainer_full, metrics_full, time_full = train_and_evaluate(cfg, cfg.OUTPUT_DIR)
    
    # Train with reduced dataset
    print("\nTraining with reduced dataset...")
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, "reduced_dataset")
    trainer_reduced, metrics_reduced, time_reduced = train_and_evaluate(cfg, cfg.OUTPUT_DIR, reduced_dataset)
    
    print("\n--- Results ---")
    print(f"Full dataset - Training time: {time_full:.2f}s, mAP: {metrics_full['bbox']['AP']:.4f}")
    print(f"Reduced dataset - Training time: {time_reduced:.2f}s, mAP: {metrics_reduced['bbox']['AP']:.4f}")
    
    # Exercise 2: Increase learning rate
    print("\n--- Exercise 2: Increase learning rate ---")
    cfg.SOLVER.BASE_LR = 0.01  # Increased learning rate
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, "increased_lr")
    print(f"Training with increased learning rate: {cfg.SOLVER.BASE_LR}")
    trainer_lr, metrics_lr, time_lr = train_and_evaluate(cfg, cfg.OUTPUT_DIR)
    print(f"Increased LR - Training time: {time_lr:.2f}s, mAP: {metrics_lr['bbox']['AP']:.4f}")
    
    # Exercise 3: Modify test threshold
    print("\n--- Exercise 3: Modify test threshold ---")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Lower threshold
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, "lower_threshold")
    print(f"Testing with lower threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
    
    # Evaluate with lower threshold
    evaluator = COCOEvaluator("balloon_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "balloon_val")
    metrics_low_thresh = inference_on_dataset(trainer_lr.model, val_loader, evaluator)
    print(f"mAP with lower threshold: {metrics_low_thresh['bbox']['AP']:.4f}")
    
    # Visualize predictions with different thresholds
    print("\nVisualizing predictions...")
    visualize_predictions(cfg, DatasetCatalog.get("balloon_val"), balloon_metadata, cfg.OUTPUT_DIR)
    
    # Exercise 4: Increase maximum iterations
    print("\n--- Exercise 4: Increase maximum iterations ---")
    cfg.SOLVER.MAX_ITER = 2000  # Double the iterations
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, "more_iterations")
    print(f"Training with {cfg.SOLVER.MAX_ITER} iterations...")
    trainer_more_iters, metrics_more_iters, time_more_iters = train_and_evaluate(cfg, cfg.OUTPUT_DIR)
    print(f"More iterations - Training time: {time_more_iters:.2f}s, mAP: {metrics_more_iters['bbox']['AP']:.4f}")
    
    # Exercise 5: Test with custom image
    print("\n--- Exercise 5: Test with custom image ---")
    # Placeholder for custom image testing
    print("To test with a custom image, please provide the image path and uncomment the code in the script.")
    """
    # Uncomment and modify this section to test with a custom image
    custom_image_path = "path_to_your_image.jpg"
    if os.path.exists(custom_image_path):
        print(f"Testing with custom image: {custom_image_path}")
        im = cv2.imread(custom_image_path)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                      metadata=balloon_metadata,
                      scale=0.8,
                      instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(OUTPUT_DIR, "custom_image_prediction.jpg")
        cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
        print(f"Saved custom image prediction to {output_path}")
    else:
        print(f"Custom image not found at {custom_image_path}")
    """
    
    print("\nAll exercises completed!")

if __name__ == "__main__":
    main()
