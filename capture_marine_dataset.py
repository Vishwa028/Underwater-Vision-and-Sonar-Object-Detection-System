import argparse
import time
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture labeled webcam images for the marine dataset.")
    parser.add_argument("label", help="Class label, for example fish or coral")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--output", type=Path, default=Path("dataset"), help="Dataset root folder")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    target_dir = args.output / args.split / args.label
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    print("Press SPACE to save a frame, Q to quit.")
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        preview = frame.copy()
        cv2.putText(preview, f"{args.label} | saved: {saved}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.imshow("Marine Dataset Capture", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            filename = target_dir / f"{args.label}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
