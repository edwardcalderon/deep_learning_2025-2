import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import time

# Register the dataset
def register_balloon_dataset(data_path, dataset_name):
    register_coco_instances(
        dataset_name,
        {},
        os.path.join(data_path, "via_region_data.json"),
        os.path.join(data_path, "")
    )
    return MetadataCatalog.get(dataset_name).set(thing_classes=["balloon"])

# Configuration setup
def setup_cfg(config_file, output_dir, base_lr=0.00025, max_iter=1000, batch_size=2):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ("balloon_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

# Training function
def train_model(cfg, dataset_name, output_dir):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Time the training
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Save training time
    with open(os.path.join(output_dir, 'training_time.txt'), 'w') as f:
        f.write(f"Training time: {training_time:.2f} seconds")
    
    return trainer, training_time

# Evaluation function
def evaluate_model(cfg, trainer, dataset_name):
    evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
    return metrics

# Visualization function
def visualize_predictions(cfg, dataset_name, output_dir, num_samples=2):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Default threshold
    predictor = DefaultPredictor(cfg)
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    for d in random.sample(dataset_dicts, num_samples):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(14, 10))
        plt.imshow(out.get_image())
        plt.axis('off')
        output_path = os.path.join(output_dir, f"prediction_{os.path.basename(d['file_name'])}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

# Main function
def main():
    # Paths
    data_dir = "balloon"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # Register datasets
    register_balloon_dataset(train_dir, "balloon_train")
    register_balloon_dataset(val_dir, "balloon_val")
    
    # Exercise 1: Baseline with full dataset
    print("=== Exercise 1: Baseline with full dataset ===")
    cfg_full = setup_cfg(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "output/full_dataset"
    )
    trainer_full, time_full = train_model(cfg_full, "balloon_val", "output/full_dataset")
    metrics_full = evaluate_model(cfg_full, trainer_full, "balloon_val")
    visualize_predictions(cfg_full, "balloon_val", "output/full_dataset")
    
    # Exercise 2: Reduce training examples by half
    print("\n=== Exercise 2: Half dataset ===")
    # Get full dataset and split in half
    dataset_dicts = DatasetCatalog.get("balloon_train")
    half_size = len(dataset_dicts) // 2
    DatasetCatalog.pop("balloon_train_half", None)
    DatasetCatalog.register("balloon_train_half", lambda: dataset_dicts[:half_size])
    MetadataCatalog.get("balloon_train_half").set(thing_classes=["balloon"])
    
    cfg_half = setup_cfg(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "output/half_dataset"
    )
    cfg_half.DATASETS.TRAIN = ("balloon_train_half",)
    trainer_half, time_half = train_model(cfg_half, "balloon_val", "output/half_dataset")
    metrics_half = evaluate_model(cfg_half, trainer_half, "balloon_val")
    
    # Exercise 3: Increase learning rate
    print("\n=== Exercise 3: Higher learning rate ===")
    cfg_high_lr = setup_cfg(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "output/high_lr",
        base_lr=0.0025  # 10x higher than default
    )
    trainer_high_lr, time_high_lr = train_model(cfg_high_lr, "balloon_val", "output/high_lr")
    metrics_high_lr = evaluate_model(cfg_high_lr, trainer_high_lr, "balloon_val")
    
    # Exercise 4: More iterations
    print("\n=== Exercise 4: More iterations ===")
    cfg_more_iters = setup_cfg(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "output/more_iters",
        max_iter=2000  # Double the iterations
    )
    trainer_more_iters, time_more_iters = train_model(cfg_more_iters, "balloon_val", "output/more_iters")
    metrics_more_iters = evaluate_model(cfg_more_iters, trainer_more_iters, "balloon_val")
    
    # Exercise 5: Test with custom threshold
    print("\n=== Exercise 5: Test with custom threshold ===")
    def test_with_threshold(cfg, dataset_name, threshold):
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        predictor = DefaultPredictor(cfg)
        
        # Get evaluation metrics with this threshold
        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, dataset_name)
        metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
        
        # Visualize predictions
        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        
        for d in random.sample(dataset_dicts, 2):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(
                im[:, :, ::-1],
                metadata=metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE_BW
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(figsize=(14, 10))
            plt.imshow(out.get_image())
            plt.axis('off')
            output_path = os.path.join(
                cfg.OUTPUT_DIR, 
                f"prediction_thresh_{threshold}_{os.path.basename(d['file_name'])}"
            )
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        
        return metrics
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    threshold_metrics = {}
    for thresh in thresholds:
        print(f"\nTesting with threshold: {thresh}")
        threshold_metrics[thresh] = test_with_threshold(
            cfg_full, "balloon_val", thresh
        )
    
    # Print summary of results
    print("\n=== Summary of Results ===")
    print(f"1. Full dataset - Time: {time_full:.2f}s, Metrics: {metrics_full}")
    print(f"2. Half dataset - Time: {time_half:.2f}s, Metrics: {metrics_half}")
    print(f"3. High LR - Time: {time_high_lr:.2f}s, Metrics: {metrics_high_lr}")
    print(f"4. More iters - Time: {time_more_iters:.2f}s, Metrics: {metrics_more_iters}")
    print("5. Threshold testing:")
    for thresh, metrics in threshold_metrics.items():
        print(f"   Threshold {thresh}: {metrics}")

if __name__ == "__main__":
    main()
