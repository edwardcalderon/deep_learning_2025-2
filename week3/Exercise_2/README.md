# YOLOv8 Study: Inference and Training (Potholes + Internet Images)

This folder contains scripts to:

- Run inference across multiple YOLOv8 pretrained models on local and internet images (`scripts/run_yolov8_inference.py`).
- Train YOLOv8 models on the Kaggle pothole segmentation dataset in Colab/Kaggle and compare results (`scripts/yolo_potholes_train_colab.py`).
- Produce an inference video from the best trained model.

## Folder Structure

- `scripts/run_yolov8_inference.py`: Local script to test multiple YOLOv8 models (`n/s/m/l/x`) on a set of images (local folder and/or downloaded from the web).
- `scripts/yolo_potholes_train_colab.py`: Script designed for Colab/Kaggle to download the Kaggle dataset, train models, compare metrics, and generate an inference video.
- `../requirements.txt`: Python dependencies for local runs.

## 1) Local Inference Across Models

Install dependencies (recommend a virtual environment):

```bash
pip install -r ../../requirements.txt
```

Run inference using a local images folder (adjust path):

```bash
python scripts/run_yolov8_inference.py \
  --images_dir "../Exercise 2/Pothole_Segmentation_YOLOv8/Pothole_Segmentation_YOLOv8/valid/images" \
  --imgsz 640 --conf 0.25 --out_dir ./inference_outputs
```

Also test with internet images (cars, motorcycles, etc.):

```bash
python scripts/run_yolov8_inference.py --download_default_urls --out_dir ./inference_outputs
```

You can add your own URLs:

```bash
python scripts/run_yolov8_inference.py --download_default_urls \
  --extra_urls https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg \
  --out_dir ./inference_outputs
```

The script will create subfolders per model and save the annotated images.

## 2) Training in Colab/Kaggle (Kaggle Pothole Dataset)

Dataset: https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset

Two supported environments:

- Colab: the script downloads the dataset via `opendatasets` (requires Kaggle credentials).
- Kaggle Notebook: attach the dataset via the sidebar ("Add data").

### Colab Steps

1. Open a Colab notebook.
2. Install packages:
   ```python
   !pip install -U ultralytics opendatasets pandas opencv-python requests
   ```
3. Provide Kaggle credentials (one of):
   - Upload `kaggle.json` to `/root/.kaggle/kaggle.json` and set permissions `600`.
   - Or set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.
4. Run the training script (adjust repo path if needed):
   ```python
   !python /content/<repo_path>/week3/Exercise_2/scripts/yolo_potholes_train_colab.py \
       --epochs 15 --imgsz 640 --project /content/runs_potholes
   ```

The script will:
- Locate/download the dataset.
- Train `yolov8n/s/m/l/x` variants.
- Save runs and `results.csv` per model under `--project`.
- Build a `comparison.csv` with mAP/precision/recall if available.
- Pick the best model and generate an inference video.

### Kaggle Notebook Steps

1. Create a new Kaggle Notebook (GPU).
2. Add Data: search and attach `farzadnekouei/pothole-image-segmentation-dataset`.
3. Install packages in a cell:
   ```python
   !pip install -U ultralytics pandas opencv-python requests
   ```
4. Run the training script (adjust path):
   ```python
   !python /kaggle/working/week3/Exercise_2/scripts/yolo_potholes_train_colab.py \
       --epochs 15 --imgsz 640 --project /kaggle/working/runs_potholes
   ```

## 3) Bonus: Try Another YOLO Dataset

- Replace the dataset in the training script by attaching/downloading a different YOLO-formatted dataset that contains `data.yaml`, `train/`, `valid/`.
- Re-run training and compare results.

## FAQ

- Why not use ffmpeg post-processing? Ultralytics already saves MP4 outputs that can be displayed directly.
- How to change models? Pass `--models yolov8n.pt yolov8s.pt ...` to either script.
- How to change confidence threshold or image size? Use `--conf` and `--imgsz` on inference; use `--imgsz` on training.

## Outputs

- Inference outputs: `./inference_outputs/`
- Training runs: `./runs_potholes/` (or your `--project` path)
- Comparison CSV: `./runs_potholes/comparison.csv`
- Inference video path: printed at the end of the training script
