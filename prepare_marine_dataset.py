import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy downloaded images into dataset/train and dataset/val folders."
    )
    parser.add_argument("source", type=Path, help="Folder containing downloaded images")
    parser.add_argument("label", help="Target class label, for example fish or coral")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Dataset root folder")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to import")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    return [path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Source folder not found: {args.source}")

    images = collect_images(args.source)
    if not images:
        raise SystemExit("No images found in the source folder.")

    random.seed(args.seed)
    random.shuffle(images)

    if args.limit > 0:
        images = images[: args.limit]

    val_count = max(1, int(len(images) * args.val_ratio)) if len(images) > 1 else 0
    val_images = images[:val_count]
    train_images = images[val_count:]

    train_dir = args.dataset / "train" / args.label
    val_dir = args.dataset / "val" / args.label
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    operation = shutil.move if args.move else shutil.copy2

    for split_name, split_images, target_dir in (
        ("train", train_images, train_dir),
        ("val", val_images, val_dir),
    ):
        for index, image_path in enumerate(split_images, start=1):
            target_name = f"{args.label}_{split_name}_{index:04d}{image_path.suffix.lower()}"
            target_path = target_dir / target_name
            operation(str(image_path), str(target_path))

    print(f"Imported label: {args.label}")
    print(f"Train images: {len(train_images)} -> {train_dir}")
    print(f"Val images: {len(val_images)} -> {val_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
