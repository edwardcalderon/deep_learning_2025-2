"""
Train YOLOv8 on a local YOLO-formatted pothole dataset (no Colab/Kaggle logic).

- Expects a local dataset directory containing: data.yaml, train/, valid/ (and optionally test/).
- Trains several YOLOv8 variants (nano to xlarge) and compares results.
- Runs inference to produce an annotated video with the best model.

Quick start (from repo root):
  python week3/Exercise_2/scripts/yolo_potholes_train_colab.py \
      --dataset_dir "week3/Exercise 2/Pothole_Segmentation_YOLOv8/Pothole_Segmentation_YOLOv8" \
      --epochs 10 --imgsz 640 --project week3/Exercise_2/runs_potholes

Note: If your dataset is a segmentation task, pass segmentation weights:
  --models yolov8n-seg.pt yolov8s-seg.pt yolov8m-seg.pt yolov8l-seg.pt yolov8x-seg.pt
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
from ultralytics import YOLO

def ensure_dataset_local(dataset_dir: Path) -> Path:
    """Validate a local dataset directory containing data.yaml, train/, valid/."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    data_yaml = dataset_dir / 'data.yaml'
    train_dir = dataset_dir / 'train'
    valid_dir = dataset_dir / 'valid'
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")
    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(f"Expected subfolders train/ and valid/ under: {dataset_dir}")
    return dataset_dir


def train_models(data_yaml: Path, models: List[str], epochs: int, imgsz: int, project: Path) -> Dict[str, Path]:
    project.mkdir(parents=True, exist_ok=True)
    runs = {}
    for m in models:
        print(f"\n=== Training {m} for {epochs} epochs, imgsz={imgsz} ===")
        model = YOLO(m)
        results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(project), name=m.replace('.pt',''))
        # Locate results CSV
        run_dir = Path(results.save_dir)
        csv = run_dir / 'results.csv'
        if not csv.exists():
            # Some versions place results.csv in parent
            alt = run_dir.parent / run_dir.name / 'results.csv'
            csv = alt if alt.exists() else csv
        runs[m] = csv
        print(f"Finished {m}. Results at: {csv}")
    return runs


def compare_results(results_csv: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for model_name, csv_path in results_csv.items():
        if not csv_path.exists():
            print(f"Warning: missing results.csv for {model_name} at {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # Take last row as best (or compute max mAP50-95 if available)
        last = df.tail(1).to_dict(orient='records')[0]
        row = {
            'model': model_name,
            'epochs_run': len(df),
            'metrics_file': str(csv_path),
        }
        # Common Ultralytics metrics columns (may vary):
        for k in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)', 'train/box_loss', 'train/cls_loss']:
            if k in df.columns:
                row[k] = float(df[k].iloc[-1])
        rows.append(row)
    comp = pd.DataFrame(rows)
    print("\nModel comparison:")
    with pd.option_context('display.max_columns', None):
        print(comp)
    return comp


def create_inference_video(trained_model_dir: Path, sample_video: Path = None) -> Path:
    """Run inference on a sample video with the best trained model and return saved video path."""
    # Find best.pt
    best = next(trained_model_dir.rglob('best.pt'))
    model = YOLO(str(best))

    if sample_video is None:
        # Try a public sample video
        import requests
        from tempfile import NamedTemporaryFile
        url = 'https://ultralytics.com/images/bus.mp4'
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        tmp = NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.write(r.content)
        tmp.flush()
        sample_video = Path(tmp.name)
        print(f"Downloaded sample video to {sample_video}")

    results = model.predict(source=str(sample_video), save=True, save_txt=False, save_conf=False)
    save_dir = Path(results[0].save_dir)
    mp4s = list(save_dir.glob('*.mp4'))
    if not mp4s:
        raise FileNotFoundError(f"No MP4 saved in {save_dir}")
    print(f"Inference video saved at: {mp4s[0]}")
    return mp4s[0]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', type=str, default='', help='Path to local YOLO dataset root (contains data.yaml).')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--models', nargs='*', default=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'])
    ap.add_argument('--project', type=str, default='./runs_potholes')
    args = ap.parse_args()

    # Resolve dataset directory
    candidate_dirs = []
    if args.dataset_dir:
        candidate_dirs.append(Path(args.dataset_dir))
    # Common local path in this repo
    candidate_dirs.append(Path('week3/Exercise 2/Pothole_Segmentation_YOLOv8/Pothole_Segmentation_YOLOv8'))
    candidate_dirs.append(Path('week3/Exercise_2/../Exercise 2/Pothole_Segmentation_YOLOv8/Pothole_Segmentation_YOLOv8'))

    dataset_dir = None
    for d in candidate_dirs:
        try:
            dataset_dir = ensure_dataset_local(d.resolve())
            break
        except Exception:
            continue
    if dataset_dir is None:
        raise FileNotFoundError("Could not locate a valid dataset directory. Pass --dataset_dir to specify it.")

    print(f"Dataset root: {dataset_dir}")

    data_yaml = dataset_dir / 'data.yaml'
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"

    results_csv = train_models(data_yaml, args.models, args.epochs, args.imgsz, Path(args.project))

    comp = compare_results(results_csv)
    # Save comparison CSV
    comp_out = Path(args.project) / 'comparison.csv'
    comp.to_csv(comp_out, index=False)
    print(f"Saved comparison: {comp_out}")

    # Pick best by mAP50-95 if available, else mAP50, else first
    best_model_name = None
    if 'metrics/mAP50-95(B)' in comp.columns:
        best_model_name = comp.sort_values('metrics/mAP50-95(B)', ascending=False).iloc[0]['model']
    elif 'metrics/mAP50(B)' in comp.columns:
        best_model_name = comp.sort_values('metrics/mAP50(B)', ascending=False).iloc[0]['model']
    else:
        best_model_name = comp.iloc[0]['model']
    print(f"Selected best model: {best_model_name}")

    best_run_dir = Path(args.project) / best_model_name.replace('.pt','')
    video_path = create_inference_video(best_run_dir)
    print(f"Final inference video: {video_path}")


if __name__ == '__main__':
    main()
