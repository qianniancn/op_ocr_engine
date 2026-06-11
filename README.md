# op_ocr_engine

`op_ocr_engine` 是一个本地 OCR HTTP 服务。当前推荐使用 `ncnn + PP-OCRv5`，用于窗口截图识别、返回文字 bbox，并给 `op` 主项目做找字和点击坐标。

服务接口保持简单：

- `GET /health`
- `GET /api/v1/version`
- `POST /api/v1/ocr`

## 1. 目录约定

推荐本地目录结构：

```text
op_ocr_engine/
  3rd_party/
    ncnn/
      x64/
        include/
        lib/
      x86/
        include/
        lib/
      arm64/
        include/
        lib/
  models/
    PP_OCRv5_mobile_det.param
    PP_OCRv5_mobile_det.bin
    PP_OCRv5_mobile_rec.param
    PP_OCRv5_mobile_rec.bin
    PP_LCNet_x0_25_textline_ori.param
    PP_LCNet_x0_25_textline_ori.bin
    ppocrv5_dict.txt
  build.py
  CMakeLists.txt
  src/
  tests/
```

这些目录不会提交到 Git：

- `3rd_party/`：本机第三方依赖。
- `build-vs*/`：CMake/Visual Studio 构建产物。
- `images/`：本地测试图片。

## 2. 第三方依赖

### 2.1 ncnn

下载地址：

- `https://github.com/Tencent/ncnn/releases`

Windows + VS2026/VS2022 推荐下载 VS2022 包：

```text
ncnn-<version>-windows-vs2022.zip
```

如果只做本地静态链接，优先使用非 `shared` 包。`shared` 包会依赖额外 DLL，交付时需要一起带上。

解压后把对应架构内容放到：

```text
op_ocr_engine/3rd_party/ncnn/x64
op_ocr_engine/3rd_party/ncnn/x86
op_ocr_engine/3rd_party/ncnn/arm64
```

CMake 默认读取：

```cmake
NCNN_ROOT = <repo>/3rd_party/ncnn
```

然后根据当前编译架构自动查找：

```text
3rd_party/ncnn/x64/lib/cmake/ncnn/ncnnConfig.cmake
3rd_party/ncnn/x86/lib/cmake/ncnn/ncnnConfig.cmake
3rd_party/ncnn/arm64/lib/cmake/ncnn/ncnnConfig.cmake
```

如果放在其他位置，可以构建时指定：

```powershell
python build.py --ncnn-root E:\path\to\ncnn
```

### 2.2 C++ HTTP/JSON 依赖

CMake 使用 `FetchContent` 拉取：

- `cpp-httplib`
- `nlohmann/json`

首次配置时需要能访问网络，或者提前准备好 CMake 的 `_deps` 缓存。

### 2.3 Tesseract 可选依赖

Tesseract 现在是可选 backend。默认 `build.py` 会关闭 Tesseract，只构建 ncnn OCR 服务。

如需 Tesseract：

```powershell
python build.py --with-tesseract
```

这时需要本机能被 CMake 找到 `Tesseract::libtesseract`。

## 3. OCR 模型

当前 ncnn backend 默认使用 PP-OCRv5 mobile 模型：

```text
PP_OCRv5_mobile_det.param
PP_OCRv5_mobile_det.bin
PP_OCRv5_mobile_rec.param
PP_OCRv5_mobile_rec.bin
PP_LCNet_x0_25_textline_ori.param
PP_LCNet_x0_25_textline_ori.bin
ppocrv5_dict.txt
```

放置路径：

```text
op_ocr_engine/models/
```

启动服务时默认按 `--model-dir models` 查找。

模型来源可以是：

- 自己从 PaddleOCR 官方模型转换为 ONNX，再转换为 ncnn。
- 使用已转换好的 PP-OCRv5 ncnn 模型。

仓库包含一套默认可用的 mobile 模型，便于拉取后直接构建和启动。替换模型时保持同名文件放在 `models/` 目录即可。

## 4. 构建

推荐使用仓库里的构建脚本：

```powershell
python build.py
```

默认行为：

```text
generator: 当前机器可用的 VS2026/VS2022
arch: x64
type: Release
target: ocr_server
Tesseract: OFF
Tests: OFF
```

常用命令：

```powershell
python build.py -g vs2026 -a x64 -t Release
python build.py -g vs2022 -a x64 -t Release
python build.py -g vs2026 -a x64 -t Release --clean
python build.py -g vs2026 -a x64 -t Release --target ocr_server
```

构建产物：

```text
build-vs2026-x64/Release/ocr_server.exe
build-vs2022-x64/Release/ocr_server.exe
```

也可以直接用 CMake：

```powershell
cmake -S . -B build-vs2026-x64 -G "Visual Studio 18 2026" -A x64 -DNCNN_ROOT=3rd_party/ncnn -DBUILD_TESSERACT_SERVER=OFF -DBUILD_TESTING=OFF
cmake --build build-vs2026-x64 --config Release --target ocr_server
```

## 5. 启动服务

ncnn mobile：

```powershell
build-vs2026-x64\Release\ocr_server.exe --backend ncnn --model-dir models --model-type mobile --host 127.0.0.1 --port 8081
```

ncnn server 模型，如果已准备对应模型文件：

```powershell
build-vs2026-x64\Release\ocr_server.exe --backend ncnn --model-dir models --model-type server --host 127.0.0.1 --port 8081
```

参数说明：

- `--backend`：`auto` / `ncnn` / `tesseract`，默认 `auto`。
- `--model-dir`：ncnn 模型目录，默认 `models`。
- `--model-type`：`mobile` 或 `server`，默认 `mobile`。
- `--use-vulkan`：启用 ncnn Vulkan，需 ncnn 包和运行环境支持。
- `--host`：监听地址，默认 `0.0.0.0`。
- `--port`：监听端口，默认 `8080`。

Tesseract backend：

```powershell
ocr_server.exe --backend tesseract --datapath tessdata --lang chi_sim --host 127.0.0.1 --port 8081
```

## 6. HTTP 协议

### 6.1 健康检查

```text
GET /health
```

响应：

```json
{ "status": "ok" }
```

### 6.2 版本信息

```text
GET /api/v1/version
```

响应示例：

```json
{ "version": "ncnn-ocr-service 1.0" }
```

### 6.3 OCR 识别

```text
POST /api/v1/ocr
Content-Type: application/json
```

请求字段：

- `image`：原始像素字节的 Base64 字符串。
- `width`：图像宽度。
- `height`：图像高度。
- `bpp`：每像素字节数，支持 `1`、`3`、`4`。

注意：接口接收的是原始像素字节，不是 PNG/JPG 文件本身。

成功响应示例：

```json
{
  "code": 0,
  "profile_ms": {
    "det": 12.3,
    "cls": 4.5,
    "rec": 30.1,
    "total": 48.2
  },
  "results": [
    {
      "text": "此电脑",
      "bbox": [10, 20, 120, 60],
      "confidence": 0.98
    }
  ]
}
```

`bbox` 是图像坐标中的外接矩形：

```text
[x1, y1, x2, y2]
```

上层如果需要点击文字，通常使用中心点：

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

## 7. Python 调用示例

```python
import base64
import json
import urllib.request
from PIL import Image

image = Image.open("input.png").convert("RGBA")
payload = {
    "width": image.width,
    "height": image.height,
    "bpp": 4,
    "image": base64.b64encode(image.tobytes()).decode("ascii"),
}

request = urllib.request.Request(
    "http://127.0.0.1:8081/api/v1/ocr",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=30) as response:
    print(response.read().decode("utf-8"))
```

## 8. 测试和可视化工具

基准测试：

```powershell
python tests\benchmark_ocr_server.py --url http://127.0.0.1:8081/api/v1/ocr --image-dir images --repeat 3 --concurrency 1
```

bbox 可视化：

```powershell
python tests\visualize_ocr_bboxes.py --url http://127.0.0.1:8081/api/v1/ocr --image-dir images --output-dir build-vs2026-x64\ocr_bbox_visuals --show-text
```

测试脚本只依赖 HTTP 接口，不参与 `ocr_server.exe` 编译。

## 9. PaddleOCR Python 服务

项目还保留了一个基于 FastAPI 的 PaddleOCR HTTP 服务：

```text
py_paddle_server/app.py
```

启动示例：

```powershell
pip install -r py_paddle_server\requirements.txt
uvicorn py_paddle_server.app:app --host 0.0.0.0 --port 8082
```

它的 HTTP 协议与 C++ 服务保持兼容，可用于和 ncnn 服务做精度对比。
