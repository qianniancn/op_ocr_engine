import argparse
import base64
import concurrent.futures
import json
import subprocess
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image


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
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return image.width, image.height, elapsed_ms, result


def summarize_result(result):
    items = result.get("results", [])
    confidences = [
        float(item.get("confidence", 0.0))
        for item in items
        if isinstance(item.get("confidence", 0.0), (int, float))
    ]
    avg_confidence = statistics.mean(confidences) if confidences else 0.0
    preview = " | ".join(str(item.get("text", "")) for item in items[:3])
    return len(items), avg_confidence, preview


def normalize_text(text: str):
    return "".join(str(text).split()).lower()


def evaluate_accuracy(image_name: str, result, expected):
    if not expected or image_name not in expected:
        return "", "", ""

    items = result.get("results", [])
    texts = [str(item.get("text", "")) for item in items]
    merged = normalize_text("".join(texts))
    spec = expected[image_name]
    expected_texts = spec.get("contains", [])
    expected_count = spec.get("count")
    inverted = spec.get("inverted")

    missing = [text for text in expected_texts if normalize_text(text) not in merged]
    count_ok = expected_count is None or expected_count == len(items)
    text_ok = not missing
    accuracy_ok = count_ok and text_ok

    inverted_ok = ""
    if inverted is not None:
        inverted_ok = "manual"

    details = []
    if not count_ok:
        details.append(f"count expected {expected_count} got {len(items)}")
    if missing:
        details.append("missing: " + " | ".join(missing))
    return str(accuracy_ok), inverted_ok, "; ".join(details)


def get_process_memory_mb(process_name: str):
    if not process_name:
        return None
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f"$p=Get-Process -Name '{process_name}' -ErrorAction SilentlyContinue | "
            "Sort-Object WorkingSet64 -Descending | Select-Object -First 1; "
            "if ($p) { [math]::Round($p.WorkingSet64 / 1MB, 2) }"
        ),
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        return float(output) if output else None
    except Exception:
        return None


def percentile(values, pct: float):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(description="Benchmark the OCR HTTP service.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8081/api/v1/ocr",
        help="OCR endpoint URL.",
    )
    parser.add_argument(
        "--image-dir",
        default=str(Path(__file__).resolve().parents[1] / "images"),
        help="Directory containing images to benchmark.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Requests per image.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent request workers.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds.")
    parser.add_argument("--expected", help="Optional JSON file with expected OCR checks.")
    parser.add_argument(
        "--process-name",
        default="ocr_server",
        help="Process name used for memory sampling. Empty disables memory sampling.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    images = iter_images(image_dir)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be greater than 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be greater than 0")

    expected = None
    if args.expected:
        expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))

    print(f"url={args.url}")
    print(f"image_dir={image_dir}")
    print(f"image_count={len(images)} repeat={args.repeat} concurrency={args.concurrency}")
    mem_before = get_process_memory_mb(args.process_name)
    if mem_before is not None:
        print(f"memory_before_mb={mem_before:.2f}")
    print()
    print(
        "image,width,height,run,count,elapsed_ms,det_ms,cls_ms,rec_ms,total_ms,"
        "avg_confidence,accuracy_ok,inverted_ok,accuracy_details,text_preview"
    )

    all_times = []
    det_times = []
    cls_times = []
    rec_times = []
    total_times = []
    per_image_times = {}
    tasks = [(image_path, run_index) for image_path in images for run_index in range(1, args.repeat + 1)]
    for image_path in images:
        per_image_times[image_path.name] = []

    def run_task(task):
        image_path, run_index = task
        width, height, elapsed_ms, result = post_ocr(args.url, image_path, args.timeout)
        return image_path, run_index, width, height, elapsed_ms, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            image_path, run_index, width, height, elapsed_ms, result = future.result()
            count, avg_confidence, preview = summarize_result(result)
            profile = result.get("profile_ms", {})
            det_ms = float(profile.get("det", 0.0))
            cls_ms = float(profile.get("cls", 0.0))
            rec_ms = float(profile.get("rec", 0.0))
            total_ms = float(profile.get("total", 0.0))
            accuracy_ok, inverted_ok, accuracy_details = evaluate_accuracy(image_path.name, result, expected)

            all_times.append(elapsed_ms)
            det_times.append(det_ms)
            cls_times.append(cls_ms)
            rec_times.append(rec_ms)
            total_times.append(total_ms)
            per_image_times[image_path.name].append(elapsed_ms)
            safe_preview = preview.replace("\n", " ").replace("\r", " ").replace('"', '""')
            safe_details = accuracy_details.replace("\n", " ").replace("\r", " ").replace('"', '""')
            print(
                f'"{image_path.name}",{width},{height},{run_index},{count},'
                f'{elapsed_ms:.2f},{det_ms:.2f},{cls_ms:.2f},{rec_ms:.2f},{total_ms:.2f},'
                f'{avg_confidence:.6f},{accuracy_ok},{inverted_ok},"{safe_details}","{safe_preview}"'
            )

    mem_after = get_process_memory_mb(args.process_name)

    print()
    print("summary")
    print(f"requests={len(all_times)}")
    print(f"avg_ms={statistics.mean(all_times):.2f}")
    print(f"min_ms={min(all_times):.2f}")
    print(f"max_ms={max(all_times):.2f}")
    print(f"p50_ms={percentile(all_times, 0.50):.2f}")
    print(f"p95_ms={percentile(all_times, 0.95):.2f}")
    print(f"avg_det_ms={statistics.mean(det_times):.2f}")
    print(f"avg_cls_ms={statistics.mean(cls_times):.2f}")
    print(f"avg_rec_ms={statistics.mean(rec_times):.2f}")
    print(f"avg_total_profile_ms={statistics.mean(total_times):.2f}")
    if mem_before is not None and mem_after is not None:
        print(f"memory_after_mb={mem_after:.2f}")
        print(f"memory_delta_mb={mem_after - mem_before:.2f}")

    print()
    print("per_image_avg")
    for image_name, times in per_image_times.items():
        print(f"{image_name},{statistics.mean(times):.2f}")


if __name__ == "__main__":
    main()
