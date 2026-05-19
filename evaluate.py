"""
evaluate.py — Evaluate a trained model on the test split.

Outputs
-------
- Console table: precision, recall, F1, mAP@0.5, mAP@0.5:0.95 per class and overall
- runs/evaluate/<name>/confusion_matrix.png
- runs/evaluate/<name>/PR_curve.png
- runs/evaluate/<name>/samples/  — up to --n-samples annotated detection images

Usage:
    python evaluate.py
    python evaluate.py --weights runs/train/gap_drop_v1/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 gap/drop detector")
    parser.add_argument(
        "--weights",
        default="runs/train/gap_drop_v1/weights/best.pt",
        help="Path to trained weights",
    )
    parser.add_argument("--data",      default="data/data.yaml", help="Dataset YAML")
    parser.add_argument("--imgsz",     type=int, default=640,    help="Inference image size")
    parser.add_argument("--conf",      type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou",       type=float, default=0.6,  help="NMS IoU threshold")
    parser.add_argument("--name",      default="test_eval",      help="Sub-folder name inside runs/evaluate/")
    parser.add_argument("--n-samples", type=int, default=20,     help="Number of example images to save")
    return parser.parse_args()


def print_results_table(metrics, class_names):
    """Pretty-print per-class and overall metrics to the console."""
    rows = []

    # metrics arrays are only populated for classes seen in the eval split
    seen = list(metrics.box.ap_class_index) if hasattr(metrics.box, "ap_class_index") else list(range(len(metrics.box.p)))
    seen_map = {cls_idx: pos for pos, cls_idx in enumerate(seen)}

    # Per-class rows
    for i, name in enumerate(class_names):
        if i not in seen_map:
            rows.append({
                "Class": name, "Precision": None, "Recall": None,
                "F1": None, "mAP@0.5": None, "mAP@0.5:0.95": None,
            })
            continue
        j = seen_map[i]
        rows.append({
            "Class":           name,
            "Precision":       round(float(metrics.box.p[j]),  4),
            "Recall":          round(float(metrics.box.r[j]),  4),
            "F1":              round(float(metrics.box.f1[j]), 4),
            "mAP@0.5":        round(float(metrics.box.ap50[j]), 4),
            "mAP@0.5:0.95":   round(float(metrics.box.ap[j]),  4),
        })

    # Mean-over-classes row
    rows.append({
        "Class":          "ALL (mean)",
        "Precision":      round(float(metrics.box.mp),    4),
        "Recall":         round(float(metrics.box.mr),    4),
        "F1":             round(float(np.mean(metrics.box.f1)), 4),
        "mAP@0.5":       round(float(metrics.box.map50),  4),
        "mAP@0.5:0.95":  round(float(metrics.box.map),   4),
    })

    df = pd.DataFrame(rows).set_index("Class")
    separator = "-" * 65
    print(f"\n{separator}")
    print("  Detection results on test set")
    print(separator)
    print(df.to_string())
    print(separator + "\n")
    return df


def save_sample_images(model, data_yaml, save_dir, n_samples, conf, iou, imgsz):
    """Run inference on test images and save annotated examples."""
    import yaml

    save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve test image directory from the YAML
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(data_yaml).parent
    test_img_dir = dataset_root / cfg.get("test", "test/images")

    if not test_img_dir.exists():
        print(f"  [warn] Test image directory not found at {test_img_dir} — skipping samples.")
        return

    image_paths = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))
    if not image_paths:
        print("  [warn] No images found in test set — skipping samples.")
        return

    # Sample evenly across the set so the saved images are representative
    indices = np.linspace(0, len(image_paths) - 1, min(n_samples, len(image_paths)), dtype=int)
    selected = [image_paths[i] for i in indices]

    print(f"  Saving {len(selected)} sample detection images to {save_dir} ...")
    for img_path in selected:
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
        annotated = results[0].plot()  # numpy array with boxes drawn
        out_path = save_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

    print(f"  Done. Samples saved to: {save_dir}")


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at '{weights_path}'. "
            "Run train.py first, or pass --weights with the correct path."
        )

    model = YOLO(str(weights_path))
    class_names = model.names  # dict {0: 'gap', 1: 'drop'}

    print(f"\nEvaluating: {weights_path}")
    print(f"Dataset   : {args.data}\n")

    # val() on the test split computes the full metric suite
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project="runs/evaluate",
        name=args.name,
        plots=True,      # writes confusion_matrix.png and PR_curve.png
        verbose=False,
    )

    # Use the directory Ultralytics actually wrote to
    save_root = Path(metrics.save_dir)
    samples_dir = save_root / "samples"

    print(f"Saving to : {save_root}\n")

    # Print and export the results table
    df = print_results_table(metrics, list(class_names.values()))
    csv_path = save_root / "metrics.csv"
    df.to_csv(csv_path)
    print(f"Metrics saved to: {csv_path}")

    # Save annotated sample images from the test set
    save_sample_images(
        model=model,
        data_yaml=args.data,
        save_dir=samples_dir,
        n_samples=args.n_samples,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )

    print("\nEvaluation complete.")
    print(f"  Confusion matrix : {save_root}/confusion_matrix.png")
    print(f"  PR curve         : {save_root}/PR_curve.png")
    print(f"  Sample detections: {samples_dir}/")
    print(f"  Metrics CSV      : {csv_path}")


if __name__ == "__main__":
    main()
