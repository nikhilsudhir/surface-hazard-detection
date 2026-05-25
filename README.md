# surface-hazard-detection

YOLOv8s object detection model for identifying surface hazards in legged robot navigation. Trained on a custom real-world RGB image dataset covering seven hazard classes. Built as a UTS Bachelor of Engineering (Honours) capstone thesis project.

---

## Hazard classes

`debris` · `gap` · `gutter` · `kerb` · `stairs` · `step` · `tree root`

---

## Results

| Metric | Value |
|--------|-------|
| Best validation mAP@0.5 | **0.856** (epoch 47) |
| Training stopped | Epoch 67 (early stopping, patience=20) |
| Hardware | NVIDIA GeForce RTX 4070 12 GB |
| Training time | ~4–5 minutes |

### Per-class test mAP@0.5

| Class | mAP@0.5 |
|-------|---------|
| Gap | 0.995 |
| Stairs | 0.828 |
| Step | 0.665 |
| Gutter | 0.665 |
| Debris | 0.197 |
| Tree root | 0.000 |
| Kerb | — (no test samples) |

Training curves, confusion matrix, and PR curve are in [docs/thesis_figures/](docs/thesis_figures/).

---

## Project layout

```
surface-hazard-detection/
├── train.py                          ← fine-tune YOLOv8s
├── evaluate.py                       ← evaluate on test set
├── predict.py                        ← inference on new images
├── requirements.txt
├── data/
│   ├── data.yaml                     ← dataset config
│   ├── train/images/  (411 images)
│   ├── train/labels/
│   ├── valid/images/  (19 images)
│   ├── valid/labels/
│   ├── test/images/   (9 images)
│   └── test/labels/
├── runs/
│   └── detect/runs/train/gap_drop_v1/
│       └── weights/
│           └── best.pt               ← best model weights (22 MB)
└── docs/
    └── thesis_figures/               ← key plots for thesis reference
        ├── results.png
        ├── confusion_matrix_normalized.png
        ├── PR_curve.png
        └── val_batch0_pred.jpg
```

---

## Dataset

Exported from Roboflow in YOLOv8 format. 411 training images (includes augmentation), 19 validation, 9 test — collected on a real outdoor urban campus environment.

- **Roboflow project:** [gap-and-drop-and-debris-detect v2](https://universe.roboflow.com/nikhils-workspace-8h7hl/gap-and-drop-and-debris-detect/dataset/2)
- **License:** CC BY 4.0

---

## Setup

**Python 3.9 – 3.12.**

```bash
pip install -r requirements.txt
```

For GPU training (CUDA 12.x), install the CUDA-enabled PyTorch build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Train

```bash
python train.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Max epochs (early stopping at patience=20) |
| `--batch` | 16 | Batch size; reduce to 8 if OOM |
| `--imgsz` | 640 | Input resolution (pixels) |
| `--name` | `gap_drop_v1` | Sub-folder name inside `runs/train/` |
| `--model` | `yolov8s.pt` | Starting checkpoint (auto-downloaded) |

Outputs saved to `runs/train/<name>/weights/best.pt`.

---

## Evaluate

```bash
python evaluate.py
```

Or with a custom weights path:

```bash
python evaluate.py --weights runs/detect/runs/train/gap_drop_v1/weights/best.pt
```

---

## Predict

```bash
# Single image
python predict.py --source path/to/image.jpg

# Folder
python predict.py --source path/to/images/

# Custom confidence threshold
python predict.py --source path/to/image.jpg --conf 0.4
```

Annotated output is saved to `runs/predict/results/`.

---

## Model details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8s |
| Pre-training | COCO (80 classes) |
| Fine-tuning classes | 7 (see above) |
| Input resolution | 640 × 640 px |
| Optimiser | AdamW, lr0=0.001, lrf=0.01 |
| Early stopping patience | 20 epochs |
| Augmentation | Mosaic, horizontal flip, HSV jitter |
