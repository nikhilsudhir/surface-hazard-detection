"""
predict.py — Run inference on a single image, a folder of images, or a video.

Annotated outputs (bounding boxes + confidence scores) are written to
runs/predict/<name>/ and also displayed if --show is passed.

Usage:
    python predict.py --source path/to/image.jpg
    python predict.py --source path/to/folder/
    python predict.py --source path/to/video.mp4 --show
    python predict.py --source path/to/folder/ --conf 0.4 --weights custom.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 gap/drop inference")
    parser.add_argument(
        "--source",
        required=True,
        help="Image file, folder of images, or video file to run inference on",
    )
    parser.add_argument(
        "--weights",
        default="runs/train/gap_drop_v1/weights/best.pt",
        help="Path to trained weights (default: best from training run)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence score to display a detection (0–1)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for Non-Maximum Suppression",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels",
    )
    parser.add_argument(
        "--name",
        default="results",
        help="Sub-folder name inside runs/predict/",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each result in a window (requires a display)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at '{weights_path}'. "
            "Run train.py first, or point --weights at your .pt file."
        )

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: '{source_path}'")

    model = YOLO(str(weights_path))

    print(f"\nRunning inference")
    print(f"  Weights : {weights_path}")
    print(f"  Source  : {source_path}")
    print(f"  Conf    : {args.conf}  |  IoU: {args.iou}  |  Size: {args.imgsz}px")

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=True,           # write annotated images to disk
        save_txt=False,      # set True to also save YOLO-format label files
        save_conf=True,      # include confidence scores in saved labels
        show=args.show,      # pop up a window per image if requested
        project="runs/predict",
        name=args.name,
        verbose=True,
    )

    # Summarise detections across all processed images
    total_detections = sum(len(r.boxes) for r in results)
    out_dir = Path("runs/predict") / args.name

    print(f"\nDone.")
    print(f"  Images processed : {len(results)}")
    print(f"  Total detections : {total_detections}")
    print(f"  Saved to         : {out_dir}/")

    # Print a per-image breakdown when there aren't too many images
    if len(results) <= 20:
        print()
        for r in results:
            img_name = Path(r.path).name
            n = len(r.boxes)
            if n == 0:
                print(f"  {img_name}: no detections")
            else:
                labels = [model.names[int(c)] for c in r.boxes.cls.cpu()]
                confs  = [f"{c:.2f}" for c in r.boxes.conf.cpu()]
                pairs  = ", ".join(f"{l}({c})" for l, c in zip(labels, confs))
                print(f"  {img_name}: {pairs}")


if __name__ == "__main__":
    main()
