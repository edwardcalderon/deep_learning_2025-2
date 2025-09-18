# run_yolov8_inference.py
# Run YOLOv8 inference across multiple pretrained models on local and internet images.
# Usage examples:
#   python run_yolov8_inference.py --images_dir "./week3/Exercise 2/Pothole_Segmentation_YOLOv8/Pothole_Segmentation_YOLOv8/valid/images" --imgsz 640 --conf 0.25
#   python run_yolov8_inference.py --download_default_urls --out_dir ./inference_outputs

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List

import requests
from ultralytics import YOLO


def download_images(urls: List[str], dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, url in enumerate(urls):
        try:
            fname = dest_dir / f"internet_{i:02d}{Path(url).suffix or '.jpg'}"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(fname, 'wb') as f:
                f.write(r.content)
            saved.append(fname)
            print(f"Downloaded: {url} -> {fname}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    return saved


def collect_images(images_dir: Path, patterns=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[Path]:
    imgs = []
    if images_dir and images_dir.is_dir():
        for p in images_dir.rglob('*'):
            if p.suffix.lower() in patterns:
                imgs.append(p)
    return imgs


def run_inference(models: List[str], images: List[Path], out_dir: Path, imgsz: int, conf: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for model_name in models:
        print(f"\n=== Running model: {model_name} ===")
        model = YOLO(model_name)
        # Batch by model API; passing list of paths works
        results = model.predict(source=[str(p) for p in images], imgsz=imgsz, conf=conf, save=True)
        # results list contains per-image outputs; the save_dir is consistent for the batch
        save_dir = Path(results[0].save_dir)
        # Copy outputs into organized folder per model
        model_out = out_dir / save_dir.name / model_name.replace('.pt', '')
        model_out.mkdir(parents=True, exist_ok=True)
        # Move/copy only annotated images (*.jpg, *.png)
        for f in save_dir.glob('*'):
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                shutil.copy2(f, model_out / f.name)
        print(f"Saved annotated outputs to: {model_out}")
        # Summarize basic info
        num_imgs = len([f for f in (model_out.iterdir()) if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
        summary.append((model_name, num_imgs, str(model_out)))

    print("\n==== Summary ====")
    for m, n, path in summary:
        print(f"Model: {m:12s} | Images saved: {n:3d} | Dir: {path}")


def default_urls() -> List[str]:
    # A small set of CC images containing cars, motorcycles, etc.
    return [
        "https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg",  # car street
        "https://images.pexels.com/photos/1453494/pexels-photo-1453494.jpeg", # motorcycles
        "https://images.pexels.com/photos/120049/pexels-photo-120049.jpeg",  # bus
        "https://images.pexels.com/photos/248747/pexels-photo-248747.jpeg",  # bicycle
        "https://images.pexels.com/photos/167962/pexels-photo-167962.jpeg",  # trucks
    ]


def parse_args():
    ap = argparse.ArgumentParser(description="Run YOLOv8 inference across multiple models on local and internet images.")
    ap.add_argument('--images_dir', type=str, default='', help='Directory with local images to test.')
    ap.add_argument('--download_default_urls', action='store_true', help='Download a small default set of internet images.')
    ap.add_argument('--extra_urls', type=str, nargs='*', default=[], help='Additional image URLs to download.')
    ap.add_argument('--out_dir', type=str, default='./inference_outputs', help='Directory to store outputs.')
    ap.add_argument('--models', type=str, nargs='*', default=[
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    ], help='List of YOLOv8 pretrained model weights to test.')
    ap.add_argument('--imgsz', type=int, default=640, help='Image size for inference.')
    ap.add_argument('--conf', type=float, default=0.25, help='Confidence threshold.')
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    images: List[Path] = []
    if args.images_dir:
        local_images = collect_images(Path(args.images_dir))
        print(f"Found {len(local_images)} local images under {args.images_dir}")
        images.extend(local_images)

    dl_urls: List[str] = []
    if args.download_default_urls:
        dl_urls.extend(default_urls())
    if args.extra_urls:
        dl_urls.extend(args.extra_urls)

    if dl_urls:
        downloaded = download_images(dl_urls, out_dir / 'internet_images')
        images.extend(downloaded)

    images = [Path(p) for p in images if Path(p).exists()]
    if not images:
        print("No images found. Provide --images_dir and/or --download_default_urls/--extra_urls.")
        sys.exit(1)

    run_inference(args.models, images, out_dir, args.imgsz, args.conf)


if __name__ == '__main__':
    main()
