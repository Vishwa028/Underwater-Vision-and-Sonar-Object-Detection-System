import argparse
import hashlib
from pathlib import Path

import requests


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download marine dataset images from a text file of URLs.")
    parser.add_argument("url_file", type=Path, help="Text file containing one image URL per line")
    parser.add_argument("output", type=Path, help="Output folder for downloaded images")
    parser.add_argument("--timeout", type=int, default=20, help="Download timeout in seconds")
    return parser.parse_args()


def infer_extension(url: str, content_type: str) -> str:
    lower_url = url.lower()
    for extension in IMAGE_EXTENSIONS:
        if lower_url.endswith(extension):
            return extension

    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "bmp" in content_type:
        return ".bmp"
    return ".jpg"


def main() -> int:
    args = parse_args()
    if not args.url_file.exists():
        raise SystemExit(f"URL file not found: {args.url_file}")

    urls = [line.strip() for line in args.url_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not urls:
        raise SystemExit("No URLs found in the input file.")

    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    downloaded = 0

    for index, url in enumerate(urls, start=1):
        try:
            response = session.get(url, timeout=args.timeout)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "image" not in content_type:
                print(f"Skipped non-image URL: {url}")
                continue

            extension = infer_extension(url, content_type)
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            filename = args.output / f"download_{index:04d}_{digest}{extension}"
            filename.write_bytes(response.content)
            downloaded += 1
            print(f"Downloaded: {filename.name}")
        except Exception as error:
            print(f"Failed: {url} | {error}")

    print(f"Finished. Downloaded {downloaded} images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
