import argparse
import base64
import json
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def iter_images(image_dir: Path):
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def post_ocr(url: str, image_path: Path, timeout: float):
    image = Image.open(image_path).convert("RGBA")
    payload = {
        "width": image.width,
        "height": image.height,
        "bpp": 4,
        "image": base64.b64encode(image.tobytes()).decode("ascii"),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return image, json.loads(response.read().decode("utf-8"))


def load_font(size: int):
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_bboxes(image: Image.Image, result, show_text: bool):
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(14, image.width // 90))
    small_font = load_font(max(12, image.width // 120))

    colors = [
        (255, 67, 67),
        (24, 144, 255),
        (82, 196, 26),
        (250, 173, 20),
        (114, 46, 209),
        (19, 194, 194),
    ]

    for index, item in enumerate(result.get("results", []), 1):
        bbox = item.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = colors[(index - 1) % len(colors)]
        line_width = max(2, image.width // 600)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = str(index)
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            label = f"{index} {confidence:.2f}"

        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 6
        label_y = max(0, y1 - label_height)
        draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height), fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

        if show_text:
            text = str(item.get("text", ""))
            if text:
                preview = text[:60]
                text_y = min(image.height - 18, y2 + 2)
                draw.text((x1, text_y), preview, fill=color, font=small_font)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Draw OCR bounding boxes over images.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8081/api/v1/ocr",
        help="OCR endpoint URL.",
    )
    parser.add_argument(
        "--image-dir",
        default=str(Path(__file__).resolve().parents[1] / "images"),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "build-vs2026-x64" / "ocr_bbox_visuals"),
        help="Directory for rendered bbox images.",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds.")
    parser.add_argument("--show-text", action="store_true", help="Draw text previews under boxes.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(image_dir)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    print(f"url={args.url}")
    print(f"image_dir={image_dir}")
    print(f"output_dir={output_dir}")
    print()

    for image_path in images:
        image, result = post_ocr(args.url, image_path, args.timeout)
        visual = draw_bboxes(image, result, args.show_text)
        output_path = output_dir / f"{image_path.stem}_bboxes.png"
        visual.save(output_path)
        print(f"{image_path.name}: count={len(result.get('results', []))} -> {output_path}")


if __name__ == "__main__":
    main()
