import argparse
import base64
import csv
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


DEFAULT_VARIANTS = "v5:mobile,v6:tiny,v6:small,v6:medium"


def parse_variants(value):
    variants = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        version, _, model_type = item.partition(":")
        if not version or not model_type:
            raise SystemExit(f"Invalid variant {item!r}; expected version:type")
        variants.append((version, model_type))
    if not variants:
        raise SystemExit("No benchmark variants were provided")
    return variants


def load_images(image_dir):
    if not image_dir:
        return [("synthetic-white-256x96", 256, 96, 1, bytes([255]) * 256 * 96)]

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required when --image-dir is used") from exc

    paths = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ]
    if not paths:
        raise SystemExit(f"No images found in {image_dir}")

    images = []
    for path in sorted(paths):
        image = Image.open(path).convert("RGBA")
        images.append((path.name, image.width, image.height, 4, image.tobytes()))
    return images


def post_json(url, payload=None, timeout=20):
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def wait_for_health(port, process, timeout):
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            status, body = post_json(url, timeout=2)
            if status == 200 and body.get("status") == "ok":
                return True
        except Exception:
            time.sleep(0.2)
    return False


def stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_variant(args, variant_index, version, model_type, images):
    port = args.start_port + variant_index
    command = [
        str(args.exe),
        "--backend", "ncnn",
        "--model-dir", str(args.model_dir),
        "--model-version", version,
        "--model-type", model_type,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--threads", str(args.threads),
    ]
    if args.use_vulkan:
        command.append("--use-vulkan")
    if args.gpu_device is not None:
        command += ["--gpu-device", str(args.gpu_device)]
    if args.fp16:
        command.append("--fp16")
    if args.int8:
        command.append("--int8")
    if args.bf16:
        command.append("--bf16")

    process = subprocess.Popen(
        command,
        cwd=args.cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        if not wait_for_health(port, process, args.startup_timeout):
            output = process.stdout.read() if process.stdout else ""
            raise SystemExit(f"{version}:{model_type} did not become healthy\n{output}")

        rows = []
        url = f"http://127.0.0.1:{port}/api/v1/ocr"
        for image_name, width, height, bpp, pixels in images:
            payload = {
                "width": width,
                "height": height,
                "bpp": bpp,
                "image": base64.b64encode(pixels).decode("ascii"),
            }
            for run in range(1, args.repeat + 1):
                start = time.perf_counter()
                status, body = post_json(url, payload, timeout=args.request_timeout)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if status != 200:
                    raise SystemExit(f"{version}:{model_type} returned HTTP {status}")
                profile = body.get("profile_ms", {})
                rows.append({
                    "variant": f"{version}:{model_type}",
                    "image": image_name,
                    "run": run,
                    "count": len(body.get("results", [])),
                    "elapsed_ms": elapsed_ms,
                    "det_ms": profile.get("det", 0.0),
                    "cls_ms": profile.get("cls", 0.0),
                    "rec_ms": profile.get("rec", 0.0),
                    "total_ms": profile.get("total", 0.0),
                })
        return rows
    finally:
        stop_process(process)


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    for variant, items in grouped.items():
        elapsed = [r["elapsed_ms"] for r in items]
        total = [float(r["total_ms"]) for r in items]
        counts = [int(r["count"]) for r in items]
        print(
            f"{variant}: requests={len(items)} "
            f"avg_elapsed={statistics.mean(elapsed):.2f}ms "
            f"p50_elapsed={statistics.median(elapsed):.2f}ms "
            f"avg_profile_total={statistics.mean(total):.2f}ms "
            f"avg_results={statistics.mean(counts):.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark op_ocr_engine ncnn model variants.")
    parser.add_argument("--exe", type=Path, default=Path("build-vs2026-x64/Release/ocr_server.exe"))
    parser.add_argument("--model-dir", type=Path, default=Path("../LiteOCR/models"))
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--use-vulkan", action="store_true")
    parser.add_argument("--gpu-device", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--start-port", type=int, default=18100)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if args.repeat <= 0:
        raise SystemExit("--repeat must be greater than 0")
    if args.start_port <= 0 or args.start_port > 65535:
        raise SystemExit("--start-port must be between 1 and 65535")

    args.cwd = Path.cwd()
    args.exe = args.exe.resolve()
    args.model_dir = args.model_dir.resolve()
    if not args.exe.exists():
        raise SystemExit(f"Server executable not found: {args.exe}")
    if not args.model_dir.exists():
        raise SystemExit(f"Model directory not found: {args.model_dir}")

    images = load_images(args.image_dir)
    variants = parse_variants(args.variants)

    all_rows = []
    for index, (version, model_type) in enumerate(variants):
        all_rows.extend(run_variant(args, index, version, model_type, images))

    summarize(all_rows)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
