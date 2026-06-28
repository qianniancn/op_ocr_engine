# op_ocr_engine

`op_ocr_engine` 是给 `op/libop` 用的本地 OCR HTTP 服务。主程序是 C++ 版 `ocr_server.exe`，用 ncnn 跑 PaddleOCR/PP-OCR 模型，负责从窗口截图里识别文字并返回文字框。

仓库只放代码、脚本和文档。OCR 模型文件比较大，不再提交到 GitHub，部署或本地调试时单独下载。

## 服务约定

`libop` 通过 `SetOcrEngine` 选择 OCR 后端。现在保留三个名字：

```text
tesseract      -> http://127.0.0.1:8080/api/v1/ocr
paddle         -> http://127.0.0.1:8081/api/v1/ocr
paddle_ncnn    -> http://127.0.0.1:8082/api/v1/ocr
```

`paddle_ncnn` 是本项目的主服务。这个名字的意思是：模型来自 PaddleOCR/PP-OCR，推理运行时是 ncnn。

`libop` 里通常这样配置：

```text
SetOcrEngine("paddle_ncnn", "", "")
```

如果想完全绕开后端名，也可以直接传 URL：

```text
SetOcrEngine("", "", "--url=http://127.0.0.1:8082/api/v1/ocr --timeout=3000")
```

端口分工不要混用：

```text
8080  Tesseract HTTP 服务
8081  py_paddle_server，Python PaddleOCR 服务
8082  ocr_server.exe，C++ ncnn 服务
```

## 本地目录

建议把运行时文件放在仓库内的本地目录里，但不要提交这些目录：

```text
op_ocr_engine/
  3rd_party/           # ncnn 预编译包，build.py 可自动准备
  build-vs2026-x64/    # 构建产物
  images/              # 本地测试图片
  models/              # OCR 模型，本机下载，不提交 GitHub
  py_paddle_server/
  src/
  tests/
  build.py
  CMakeLists.txt
```

启动服务时建议显式传模型目录：

```powershell
--model-dir models
```

这样不依赖当前工作目录，也不依赖程序里保留的默认路径。

## 模型准备

模型下载：

[https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/](https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/)

也可以加 QQ 群 `743710486` 获取模型文件。

下载后放到本地 `models/` 目录。默认模型是 `small`。

PP-OCRv6 `tiny` 适合只关心速度的场景，比如快速扫按钮、菜单、标题这类界面文字。它启动和推理都比较轻，但遇到小字号、低对比度或密集文字时，稳定性不如 `small`。

`tiny` 必需文件：

```text
models/
  PP-OCRv6_tiny_det.param
  PP-OCRv6_tiny_det.bin
  PP-OCRv6_tiny_rec.param
  PP-OCRv6_tiny_rec.bin
  PP-OCRv6_vocab_tiny.txt
```

PP-OCRv6 `small` 是默认选择，适合 `libop` 日常找字、点按钮、读窗口文本。它比 `tiny` 稳一些，又不会像 `medium` 那样明显增加耗时，普通使用先选它。

`small` 必需文件：

```text
models/
  PP-OCRv6_small_det.param
  PP-OCRv6_small_det.bin
  PP-OCRv6_small_rec.param
  PP-OCRv6_small_rec.bin
  PP-OCRv6_vocab.txt
```

PP-OCRv6 `medium` 适合更看重识别质量的截图，比如小字比较多、背景复杂、文字挤在一起的页面。代价是模型更大，推理会比 `small` 慢一些。

`medium` 必需文件：

```text
models/
  PP-OCRv6_medium_det.param
  PP-OCRv6_medium_det.bin
  PP-OCRv6_medium_rec.param
  PP-OCRv6_medium_rec.bin
  PP-OCRv6_vocab.txt
```

方向分类是通用可选项，用来处理倒置文字。常规窗口截图一般可以不放；如果遇到文字方向不稳定，再加这组模型。

可放：

```text
models/
  PP-LCNet_x1_0_textline_ori.param
  PP-LCNet_x1_0_textline_ori.bin
```

表格结构识别不是当前 `ocr_server.exe` 的必需能力。只做文字识别、找字和点击坐标时，不需要下面这些文件。

如果以后要把截图里的表格还原成 HTML 结构，才需要表格模型：

```text
models/
  PP-StructrureV2_SLANet_plus_cnn.param
  PP-StructrureV2_SLANet_plus_cnn.bin
  PP-StructrureV2_SLANet_plus_slahead.param
  PP-StructrureV2_SLANet_plus_slahead.bin
  table_structure_dict_ch.txt
```

如果只是识别表格单元格里的文字，仍然用 `tiny`、`small` 或 `medium`，不需要表格结构模型。

PP-OCRv5 主要用于和历史模型结果做对比，或者在需要复现 v5 行为时使用。新场景优先用 PP-OCRv6。

PP-OCRv5 不使用 `--quality`，需要显式指定版本和类型：

```powershell
ocr_server.exe --model-dir models --model-version v5 --model-type mobile
ocr_server.exe --model-dir models --model-version v5 --model-type server
```

PP-OCRv5 `mobile` 模型小，适合做兼容性对比或轻量回归：

```text
models/
  PP-OCRv5_mobile_det.param
  PP-OCRv5_mobile_det.bin
  PP-OCRv5_mobile_rec.param
  PP-OCRv5_mobile_rec.bin
  PP-OCRv5_vocab.txt
```

PP-OCRv5 `server` 模型更大，适合对比 v5 的高精度版本：

```text
models/
  PP-OCRv5_server_det.param
  PP-OCRv5_server_det.bin
  PP-OCRv5_server_rec.param
  PP-OCRv5_server_rec.bin
  PP-OCRv5_vocab.txt
```

注意：`build.py` 自动下载的是 ncnn 运行库，不是 OCR 模型。模型需要按上面的地址单独准备。

## 构建

Windows 下直接用构建脚本：

```powershell
python build.py
```

默认构建 C++ ncnn 服务：

```text
Generator  自动检测 VS2026 / VS2022
Arch       x64
Type       Release
Target     ocr_server
Tesseract  OFF
Tests      OFF
```

`build.py` 会检查 `3rd_party/ncnn`。如果缺少 ncnn，会下载默认的 Windows VS2022 预编译包：

```text
ncnn-20260526-windows-vs2022.zip
```

常用命令：

```powershell
python build.py -g vs2026 -a x64 -t Release
python build.py -g vs2022 -a x64 -t Release
python build.py -g vs2026 -a x64 -t Release --clean
python build.py --no-ncnn-download
python build.py --ncnn-root E:\path\to\ncnn
```

构建完成后，常用产物路径是：

```text
build-vs2026-x64\Release\ocr_server.exe
build-vs2022-x64\Release\ocr_server.exe
```

Tesseract 是单独的 OCR 后端，默认不参与构建。需要时再打开：

```powershell
python build.py --with-tesseract
```

这要求本机 CMake 能找到 `Tesseract::libtesseract`。

## 启动

日常启动默认使用 `small`：

```powershell
ocr_server.exe --model-dir models
```

也可以显式指定 PP-OCRv6 模型规格：

```powershell
ocr_server.exe --model-dir models --model-type tiny
ocr_server.exe --model-dir models --model-type small
ocr_server.exe --model-dir models --model-type medium
```

PP-OCRv5 用的是 `mobile` 和 `server` 两个规格，启动时写清楚版本：

```powershell
ocr_server.exe --model-dir models --model-version v5 --model-type mobile
ocr_server.exe --model-dir models --model-version v5 --model-type server
```

GPU 模式：

```powershell
ocr_server.exe --model-dir models --model-type small --device gpu --gpu-device 0
```

CPU 线程数：

```powershell
ocr_server.exe --model-dir models --model-type small --threads 8
```

快速模式：

```powershell
ocr_server.exe --model-dir models --model-type small --threads 8 --fast-mode
```

默认监听：

```text
http://127.0.0.1:8082
```

常用参数：

```text
--model-dir      模型目录，建议显式传 models
--model-version  模型版本，v6 或 v5，默认 v6
--model-type     v6 用 tiny / small / medium，默认 small；v5 用 mobile / server
--device         cpu / gpu，默认 cpu
--threads        ncnn CPU 推理线程数，默认 4
--fast-mode      跳过方向分类和 180 度回退识别，速度更快，但倒置文字鲁棒性会下降
--host           默认 127.0.0.1
--port           默认 8082
```

`--fast-mode` 和 `--quality fast` 不一样。`--quality fast` 是兼容别名，等价于选择 v6 `tiny` 模型；`--fast-mode` 不换模型，只是在识别流程里跳过方向分类模型和低置信度时的 180 度回退识别。

GUI 工具启动服务时也使用这些参数：模型版本传 `--model-version`，模型规格传 `--model-type`，线程档位传 `--threads`，勾选快速模式时额外传 `--fast-mode`。

调优和兼容参数：

```text
--quality       兼容别名：fast / balanced / accurate，对应 v6 tiny / small / medium；不要和 --model-version / --model-type 混用
--use-vulkan    启用 ncnn Vulkan，等价于 --device gpu
--gpu-device    Vulkan GPU 设备编号，GPU 模式默认 0
--fp16          启用 ncnn FP16 arithmetic / storage / packing
--bf16          启用 ncnn BF16 storage / packing，适合做性能和精度对比
--backend       auto / ncnn / tesseract，默认 auto
--datapath      Tesseract tessdata 目录，仅 --backend tesseract 需要
--lang          Tesseract 语言，默认 eng
```

开发调试参数可以看：

```powershell
ocr_server.exe --help-advanced
```

## HTTP 接口

服务提供三个接口：

```text
GET  /health
GET  /api/v1/version
POST /api/v1/ocr
```

健康检查：

```text
GET /health
```

响应：

```json
{ "status": "ok" }
```

OCR 请求使用 JSON，传的是原始像素字节的 Base64，不是 PNG/JPG 文件内容。

```text
POST /api/v1/ocr
Content-Type: application/json
```

请求字段：

```text
image   原始像素字节的 Base64 字符串
width   图像宽度
height  图像高度
bpp     每像素字节数，支持 1 / 3 / 4
```

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

`bbox` 是文字外接矩形：

```text
[x1, y1, x2, y2]
```

上层要点击文字时，一般取中心点：

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

## Python 调用示例

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
    "http://127.0.0.1:8082/api/v1/ocr",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=30) as response:
    print(response.read().decode("utf-8"))
```

## 测试工具

接口压测：

```powershell
python tests\benchmark_ocr_server.py --url http://127.0.0.1:8082/api/v1/ocr --image-dir images --repeat 3 --concurrency 1
```

不同模型档位对比：

```powershell
python tests\benchmark_ncnn_models.py --model-dir models --image-dir images --repeat 3 --csv build-vs2026-x64\ncnn_model_benchmark.csv
```

GPU 对比：

```powershell
python tests\benchmark_ncnn_models.py --model-dir models --use-vulkan --gpu-device 0 --threads 4
```

bbox 可视化：

```powershell
python tests\visualize_ocr_bboxes.py --url http://127.0.0.1:8082/api/v1/ocr --image-dir images --output-dir build-vs2026-x64\ocr_bbox_visuals --show-text
```

## Python PaddleOCR 服务

`py_paddle_server/` 是保留下来的 Python PaddleOCR 服务，端口固定为 `8081`。它和 C++ 服务使用同一套 HTTP 协议，主要用于精度对比和回退验证。

启动：

```powershell
pip install -r py_paddle_server\requirements.txt
uvicorn py_paddle_server.app:app --host 0.0.0.0 --port 8081
```

`libop` 切到 Python PaddleOCR：

```text
SetOcrEngine("paddle", "", "")
```
