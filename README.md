# robodogThesis — YOLOv8 Gap & Drop Detector

Fine-tuned YOLOv8 object detection model for identifying surface hazards (gaps and drops) in legged robot navigation. Built for a thesis project using the Unitree GO2 robot.

---

## Project layout

```
robodogThesis/
├── data/
│   ├── train/images/      ← training images
│   ├── train/labels/      ← training YOLO labels (.txt)
│   ├── val/images/        ← validation images
│   ├── val/labels/
│   ├── test/images/       ← held-out test images
│   ├── test/labels/
│   └── data.yaml          ← dataset config (paths + class names)
├── runs/
│   ├── train/             ← training artefacts (weights, metric plots)
│   ├── evaluate/          ← confusion matrix, PR curves, sample images
│   └── predict/           ← annotated inference output
├── train.py               ← fine-tune YOLOv8
├── evaluate.py            ← evaluate on test set
├── predict.py             ← inference on new images
└── requirements.txt
```

---

## 1 — Install dependencies

### CPU (any machine)

```bash
pip install -r requirements.txt
```

### GPU (CUDA 12.x — recommended for training)

Replace the `torch` and `torchvision` lines with the CUDA build before installing:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Python version:** 3.9 – 3.12 tested.

---

## 2 — Drop in your Roboflow dataset

1. Export your dataset from Roboflow → **YOLOv8 format**.
2. Unzip the export. You'll get `train/`, `valid/`, `test/`, and `data.yaml`.
3. Copy the folders into `data/` (rename `valid/` → `val/` if needed):

```
data/train/images/   ← ~600 images
data/train/labels/   ← matching .txt files
data/val/images/     ← ~100 images
data/val/labels/
data/test/images/    ← ~75 images
data/test/labels/
```

4. Replace `data/data.yaml` with the one from the Roboflow export **or** verify that the class names and order match:

```yaml
names:
  0: gap
  1: drop
```

---

## 3 — Train

```bash
python train.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Max training epochs (early stopping at patience=20) |
| `--batch` | 16 | Batch size; reduce to 8 if you hit OOM |
| `--imgsz` | 640 | Input resolution in pixels |
| `--name` | `gap_drop_v1` | Sub-folder name inside `runs/train/` |
| `--model` | `yolov8s.pt` | Starting checkpoint (downloaded automatically) |

Training produces:

```
runs/train/gap_drop_v1/
├── weights/
│   ├── best.pt     ← best mAP checkpoint  ← use this for eval/predict
│   └── last.pt     ← final epoch checkpoint
├── results.csv     ← epoch-by-epoch metrics
├── results.png     ← metric curves plot
├── confusion_matrix.png
└── PR_curve.png
```

---

## 4 — Evaluate on the test set

```bash
python evaluate.py
```

Pass `--weights` if your run has a different name:

```bash
python evaluate.py --weights runs/train/gap_drop_v1/weights/best.pt
```

Output printed to console:

```
-----------------------------------------------------------------
  Detection results on test set
-----------------------------------------------------------------
             Precision  Recall    F1  mAP@0.5  mAP@0.5:0.95
Class
gap              0.89    0.86  0.875     0.91          0.63
drop             0.92    0.88  0.900     0.93          0.67
ALL (mean)       0.91    0.87  0.888     0.92          0.65
-----------------------------------------------------------------
```

Saved artefacts:

```
runs/evaluate/test_eval/
├── confusion_matrix.png     ← true/false positive breakdown per class
├── PR_curve.png             ← precision–recall curves per class
├── metrics.csv              ← same table as above, machine-readable
└── samples/                 ← up to 20 annotated test images
```

---

## 5 — Predict on new images

```bash
# Single image
python predict.py --source path/to/photo.jpg

# Folder of images
python predict.py --source path/to/images/

# With a custom confidence threshold
python predict.py --source path/to/photo.jpg --conf 0.4
```

Annotated images are saved to `runs/predict/results/`. Add `--show` to display results in a pop-up window.

---

## 6 — Output file reference

| File | What it contains |
|------|-----------------|
| `best.pt` | Best model weights by validation mAP — use for eval and deployment |
| `last.pt` | Weights from the final epoch — useful if early stopping triggered too soon |
| `results.csv` | Epoch-level: box loss, cls loss, precision, recall, mAP@0.5, mAP@0.5:0.95 |
| `results.png` | Visual plot of the above curves |
| `confusion_matrix.png` | How often each class was correctly/incorrectly predicted |
| `PR_curve.png` | Precision–recall trade-off per class at varying confidence thresholds |
| `metrics.csv` | Final test-set metrics table exported by evaluate.py |
| `samples/` | Annotated test images showing model predictions in context |

---

## Model details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8s |
| Pre-training | COCO (80 classes) |
| Fine-tuning classes | gap (0), drop (1) |
| Input resolution | 640 × 640 px |
| Optimiser | AdamW |
| Early stopping patience | 20 epochs |
