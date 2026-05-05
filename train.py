"""
train.py — Fine-tune YOLOv8s on the gap/drop detection dataset.

Usage:
    python train.py
    python train.py --epochs 150 --batch 8 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 gap/drop detector")
    parser.add_argument("--model",  default="yolov8s.pt",       help="Pre-trained weights to start from")
    parser.add_argument("--data",   default="data/data.yaml",   help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=100,      help="Number of training epochs")
    parser.add_argument("--batch",  type=int, default=16,        help="Batch size (-1 = auto)")
    parser.add_argument("--imgsz",  type=int, default=640,      help="Input image size (pixels)")
    parser.add_argument("--name",   default="gap_drop_v1",      help="Run name inside runs/train/")
    parser.add_argument("--workers",type=int, default=4,        help="Dataloader worker threads")
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify the dataset YAML exists before starting
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at '{data_path}'. "
            "Drop your Roboflow export into data/ and check the path."
        )

    # Load YOLOv8s pre-trained on COCO.
    # The weights file is downloaded automatically on first run (~22 MB).
    model = YOLO(args.model)

    print(f"\n{'='*60}")
    print(f"  Training YOLOv8 gap/drop detector")
    print(f"  Weights : {args.model}")
    print(f"  Dataset : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Img size: {args.imgsz}px")
    print(f"  Batch   : {args.batch}")
    print(f"{'='*60}\n")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        name=args.name,
        project="runs/train",

        # Optimiser — AdamW works well for small fine-tuning jobs
        optimizer="AdamW",
        lr0=0.001,           # initial learning rate
        lrf=0.01,            # final lr = lr0 * lrf
        weight_decay=0.0005,

        # Early stopping: halt if mAP@0.5 doesn't improve for 20 epochs
        patience=20,

        # Data augmentation — standard flips + light colour jitter
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,          # mosaic augmentation helps small datasets

        # Logging and saving
        save=True,           # save best.pt and last.pt
        save_period=-1,      # only save best; set to N to save every N epochs
        plots=True,          # generate metric curve plots in the run directory
        verbose=True,
    )

    # Report where the artefacts landed
    best_weights = Path("runs/train") / args.name / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights saved to: {best_weights}")
    print(f"Metric plots saved to: runs/train/{args.name}/")
    print(f"\nNext step — evaluate on the test set:")
    print(f"  python evaluate.py --weights {best_weights}")


if __name__ == "__main__":
    main()
