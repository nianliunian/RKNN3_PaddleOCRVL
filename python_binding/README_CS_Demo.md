# PaddleOCR-VL 服务端推理 Demo (Client-Server 模式)

基于 RKNN3 的 PaddleOCR-VL C/S 架构推理示例。服务端部署在 RK3588 开发板上，客户端通过 HTTP 接口发送图片，获取 OCR 识别结果。

支持两种推理模式：

- **直跑模式**（`/ocr`、`/ocr_base64`、`/ocr_pdf`）：整图直接送 VLM，不切 block
- **Layout 模式**（`/ocr_layout_image`、`/ocr_layout_pdf`）：先 PP-DocLayoutV2 检出 block，再按 block 类型逐块送 VLM，最后聚合成 markdown/html

## 文件说明

```
python_binding/
├── build/build_local_rk3588_Release/
│   └── paddleocr_vl.cpython-311-aarch64-linux-gnu.so  # pybind11 编译的推理库
├── paddleocr_vl_server.py   # 服务端 — 加载模型，提供 HTTP API
├── paddleocr_vl_client.py   # 客户端 — 发送图片/PDF，显示并保存结果
├── test_paddleocr_vl_lib.py # 本地单图推理测试
├── demo.py                  # 本地多模式推理 demo
├── run_pipeline.py          # 非 HTTP pipeline CLI（单图/PDF）
└── pipeline/                # DocLayout + VLM 流水线模块
```

模型文件目录：`/root/demo_paddleocr_vl/model/`
DocLayout ONNX：`/root/demo_paddleocr_vl/PP_onnx/PP-DocLayoutV2.onnx`

## 环境依赖

- Python 3.11
- Flask（`pip3 install flask`）
- requests（客户端，开发板已预装）

## 一、启动服务端

```bash
cd /root/demo_paddleocr_vl/rknn3-model-zoo/examples/paddleocr_vl/python_binding

# 前台启动（可看日志）
python3 paddleocr_vl_server.py --port 8080

# 后台启动
nohup python3 paddleocr_vl_server.py --port 8080 > /tmp/paddleocr_vl_server.log 2>&1 &
```

服务端启动参数：

| 参数       | 默认值       | 说明               |
|------------|-------------|--------------------|
| `--host`   | `0.0.0.0`   | 监听地址           |
| `--port`   | `8080`      | 监听端口           |

启动后模型会自动加载，日志出现 `Running on http://192.168.46.102:8080` 即可使用。

## 二、API 接口

服务端提供以下 HTTP 路由：

| 路由 | 用途 | 输入 |
|---|---|---|
| `POST /ocr` | 单图直跑 VLM（不切 block） | image + prompt + prompt_type |
| `POST /ocr_base64` | 单图 base64 直跑 VLM | JSON body |
| `POST /ocr_pdf` | PDF 逐页渲染 → 每页直跑 VLM | file + dpi + prompt_type |
| `POST /ocr_layout_image` | 单图走 DocLayout → 切 block → 逐 block VLM | image + layout_threshold + visualize |
| `POST /ocr_layout_pdf` | PDF 逐页渲染 → 每页走 layout_image | file + dpi + pages + layout_threshold + visualize |
| `GET /health` | 健康检查 | — |

> **单进程单实例约束**：`paddleocr_vl.PaddleOCRVL()` 在同一 Python 进程内不能创建两个实例。服务端 `init_pipeline()` 复用 `/ocr` 路由的 model 单例，因此 `/ocr_layout_*` 路由不能与 `/ocr` 各建各的实例。

### 1. `POST /ocr` — 单图直跑 VLM

**请求：**

- `image`（file，必填）：图片文件
- `prompt`（string，可选，默认 `OCR:`）：自定义 prompt 文本
- `prompt_type`（string，可选，默认 `ocr`）：预设 prompt 类型

**响应（JSON）：**

```json
{
  "text": "识别出的文字内容",
  "metrics": {
    "vision_latency": 519.05,
    "prefill_latency": 24.40,
    "decode_latency": 1562.05,
    "ttft": 24.39,
    "prefill_tokens": 338,
    "decode_tokens": 368,
    "tps": 0.45,
    "tpt": 2.37,
    "prefill_tps": 13855.30,
    "prefill_tpt": 0.07,
    "decode_tps": 235.59,
    "decode_tpt": 4.24
  }
}
```

**curl 示例：**

```bash
curl -X POST http://192.168.46.102:8080/ocr \
  -F "image=@/root/demo_paddleocr_vl/model/test.png" \
  -F "prompt_type=ocr"
```

### 2. `POST /ocr_base64` — Base64 编码图片

**请求（JSON body）：**

```json
{
  "image": "<base64编码的图片数据>",
  "prompt": "OCR:",
  "prompt_type": "ocr"
}
```

**curl 示例：**

```bash
base64_data=$(base64 -w 0 /root/demo_paddleocr_vl/model/test.png)
curl -X POST http://192.168.46.102:8080/ocr_base64 \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$base64_data\", \"prompt_type\": \"ocr\"}"
```

### 3. `POST /ocr_pdf` — PDF 逐页直跑 VLM

不切 block，每页整图直接送 VLM。

**请求（multipart form）：**

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `file` | file | 是 | — | PDF 文件（必须 `.pdf`） |
| `prompt` | string | 否 | `OCR:` | 自定义 prompt |
| `prompt_type` | string | 否 | `ocr` | 预设类型 |
| `dpi` | int | 否 | `200` | 渲染 DPI |

> 受 `MAX_PDF_PAGES=10` 限制，超出时仅处理前 10 页，响应含 `"truncated": true`。

**响应（JSON）：**

```json
{
  "pages": [
    {"page": 1, "text": "...", "markdown": "## Page 1\n\n..."},
    ...
  ],
  "markdown": "## Page 1\n\n...\n\n---\n\n## Page 2\n\n...",
  "total_pages": 3,
  "metrics": { ... },
  "truncated": false
}
```

**curl 示例：**

```bash
curl -X POST http://192.168.46.102:8080/ocr_pdf \
  -F "file=@/path/to/doc.pdf" \
  -F "prompt_type=ocr" \
  -F "dpi=200"
```

### 4. `POST /ocr_layout_image` — 单图 Layout OCR

DocLayout 检出 block → 按 block 类型（text/table/chart/formula 等）逐块送 VLM → 聚合成 markdown/html。

**请求（multipart form）：**

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `image` | file | 是 | — | 图片文件 |
| `layout_threshold` | float | 否 | `0.5` | DocLayout score 阈值，低于此分的 box 丢弃 |
| `visualize` | string | 否 | `false` | 是否返回可视化图（`"true"`/`"false"`） |

**响应（JSON）：**

```json
{
  "text": "...",
  "markdown": "...",
  "html": "...",
  "blocks": [
    {"label": "text", "bbox": [xmin,ymin,xmax,ymax], "content": "..."},
    {"label": "table", "bbox": [...], "content": "<markdown table>"},
    ...
  ],
  "layout_det_res": {"boxes": [{"label","coordinate","score"}, ...]},
  "metrics": {
    "layout_ms": 2558.0,
    "layout_boxes": 25,
    "vlm_blocks_run": 25,
    "vlm_blocks_skipped": 0,
    "vision_latency_total_ms": 13065.5,
    "prefill_latency_total_ms": 1534.9,
    "prefill_tokens_total": 8450,
    "decode_latency_total_ms": 1636.4,
    "decode_tokens_total": 354,
    "vlm_total_ms": 16236.7,
    "page_total_ms": 18794.8,
    "vlm_tps_avg": 216.3
  },
  "visualization_b64": "..."  // 仅 visualize=true 时返回
}
```

**curl 示例：**

```bash
curl -X POST http://192.168.46.102:8080/ocr_layout_image \
  -F "image=@/path/to/page.png" \
  -F "layout_threshold=0.5" \
  -F "visualize=true"
```

### 5. `POST /ocr_layout_pdf` — PDF Layout OCR

逐页渲染 → 每页走 `ocr_layout_image` 流程。

**请求（multipart form）：**

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `file` | file | 是 | — | PDF 文件（必须 `.pdf`） |
| `layout_threshold` | float | 否 | `0.5` | DocLayout score 阈值 |
| `dpi` | int | 否 | `200` | PDF 渲染 DPI |
| `pages` | string | 否 | 全部（≤10 页） | 页码选择，1-based 闭区间 |
| `visualize` | string | 否 | `false` | 是否返回每页可视化图 |

**`pages` 参数格式：**

- 1-based 闭区间，逗号分隔
- 示例：
  - `"1-5"` → 第 1~5 页
  - `"3,7,9"` → 第 3、7、9 页
  - `"1-3,5,8-10"` → 第 1、2、3、5、8、9、10 页
  - 空 / 不传 → 所有页（受 `MAX_PDF_PAGES=10` 限制）
- 超出文档范围的页码忽略；范围 `a-b` 若 `a>b` 返回 400

**响应（JSON）：**

```json
{
  "pages": [
    {
      "page": 1,
      "text": "...",
      "markdown": "...",
      "html": "...",
      "blocks": [...],
      "layout_det_res": {"boxes": [...]},
      "metrics": {...},
      "visualization_b64": "..."
    },
    ...
  ],
  "markdown": "## Page 1\n\n...\n\n---\n\n## Page 2\n\n...",
  "total_pages_requested": 3,
  "total_pages_in_doc": 10,
  "truncated": false
}
```

**curl 示例：**

```bash
curl -X POST http://192.168.46.102:8080/ocr_layout_pdf \
  -F "file=@/path/to/doc.pdf" \
  -F "pages=1-3,5" \
  -F "dpi=200" \
  -F "visualize=true"
```

### 6. `GET /health` — 健康检查

```bash
curl http://192.168.46.102:8080/health
# 返回：{"status":"ok","model_dir":"/root/demo_paddleocr_vl/model"}
```

## 三、Prompt 参数说明

客户端可通过两种方式指定 prompt，优先级：`prompt_type` > `prompt`。

### prompt_type 预设类型

| prompt_type   | 实际 prompt 文本            | 用途         |
|---------------|-----------------------------|-------------|
| `ocr`         | `OCR:`                      | 文字识别（默认） |
| `table`       | `Table Recognition:`        | 表格识别     |
| `chart`       | `Chart Recognition:`        | 图表识别     |
| `formula`     | `Formula Recognition:`      | 公式识别     |
| `image`       | `Image Recognition:`        | 图像描述     |

当指定了有效的 `prompt_type` 时，`prompt` 参数会被忽略，使用预设模板。

### 自定义 prompt

若 `prompt_type` 不在预设列表中，则使用 `prompt` 字段的自定义文本：

```bash
python3 paddleocr_vl_client.py --image test.png --prompt "请识别图中的文字并翻译为英文"
```

## 四、客户端使用

客户端 `paddleocr_vl_client.py` 提供四种调用模式，互斥选择其一：

| 参数 | 调用路由 | 说明 |
|---|---|---|
| `--image PATH` | `/ocr` | 单图直跑 VLM |
| `--pdf PATH` | `/ocr_pdf` | PDF 逐页直跑 VLM |
| `--layout_image PATH` | `/ocr_layout_image` | 单图 Layout OCR |
| `--layout_pdf PATH` | `/ocr_layout_pdf` | PDF Layout OCR |

### 完整参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server` | `http://192.168.46.104:8080` | 服务端地址 |
| `--image` | — | 图片路径（用 `/ocr`） |
| `--pdf` | — | PDF 路径（用 `/ocr_pdf`） |
| `--layout_image` | — | 图片路径（用 `/ocr_layout_image`） |
| `--layout_pdf` | — | PDF 路径（用 `/ocr_layout_pdf`） |
| `--prompt` | `OCR:` | 自定义 prompt 文本（仅 `/ocr`、`/ocr_pdf`） |
| `--prompt_type` | `ocr` | 预设 prompt 类型（仅 `/ocr`、`/ocr_pdf`） |
| `--pdf-dpi` | `200` | PDF 渲染 DPI（`--pdf` / `--layout_pdf`） |
| `--layout_threshold` | `0.5` | DocLayout score 阈值（`--layout_*`） |
| `--pages` | `None`（全部） | PDF 页码，如 `"1-3,5"`（`--layout_pdf`） |
| `--visualize` | `False` | 请求可视化图（`--layout_*`） |
| `--output_dir` | `None`（不存盘） | 结果保存目录（`--layout_*`） |

### 输出文件保存

仅 `--layout_image` 和 `--layout_pdf` 模式支持 `--output_dir` 保存。设置后写入以下文件：

**`--layout_image`** （`--output_dir DIR`，文件名基于输入图片名 `<base>`）：

| 文件 | 条件 |
|---|---|
| `<base>.md` | 总是 |
| `<base>.json` | 总是（完整响应，含 html/blocks/metrics） |
| `<base>.html` | 响应含 html 时 |
| `<base>_vis.png` | `--visualize` 设置时 |

**`--layout_pdf`** （`--output_dir DIR`，文件名基于输入 PDF 名 `<base>`）：

| 文件 | 条件 |
|---|---|
| `<base>.md` | 总是（合并所有页） |
| `<base>.json` | 总是（完整响应） |
| `<base>.html` | 至少一页有 html 时（页间用 `<hr>` 分隔） |
| `<base>_page<N>_vis.png` | `--visualize` 设置时，按页保存 |

**示例：**

```bash
# 单图 Layout OCR，保存到 ./output
python3 paddleocr_vl_client.py \
  --server http://192.168.46.104:8080 \
  --layout_image /root/demo_paddleocr_vl/pdf_process/output_pdf/page0_scale2.0_pymupdf.png \
  --output_dir ./output \
  --visualize

# 输出：
# [client] Saved: ./output/page0_scale2.0_pymupdf.md
# [client] Saved: ./output/page0_scale2.0_pymupdf.json
# [client] Saved: ./output/page0_scale2.0_pymupdf.html
# [client] Saved: ./output/page0_scale2.0_pymupdf_vis.png

# PDF Layout OCR，前 3 页，保存到 ./output
python3 paddleocr_vl_client.py \
  --server http://192.168.46.104:8080 \
  --layout_pdf /path/to/doc.pdf \
  --pages "1-3" \
  --output_dir ./output \
  --visualize
```

### 不同模式示例

```bash
# 文字识别（单图直跑）
python3 paddleocr_vl_client.py --image test.png --prompt_type ocr

# 表格识别
python3 paddleocr_vl_client.py --image test.png --prompt_type table

# 图表识别
python3 paddleocr_vl_client.py --image test.png --prompt_type chart

# 公式识别
python3 paddleocr_vl_client.py --image test.png --prompt_type formula

# 图像描述
python3 paddleocr_vl_client.py --image test.png --prompt_type image
```

### 自定义 prompt 示例

```bash
python3 paddleocr_vl_client.py --image test.png --prompt "请将图中的中文翻译为英文"
```

## 五、性能指标说明

`/ocr`、`/ocr_base64`、`/ocr_pdf` 返回单次推理的 `metrics`：

| 指标               | 单位    | 说明                          |
|--------------------|--------|-------------------------------|
| `vision_latency`   | ms     | 视觉模型推理耗时               |
| `prefill_latency`  | ms     | Prefill 阶段耗时（首token前）   |
| `decode_latency`   | ms     | Decode 阶段耗时（生成token）    |
| `ttft`             | ms     | Time To First Token            |
| `prefill_tokens`   | 个     | Prefill 阶段 token 数           |
| `decode_tokens`    | 个     | Decode 阶段 token 数            |
| `tps`              | tok/s  | 整体吞吐率                     |
| `tpt`              | ms/tok | 整体每 token 耗时               |
| `prefill_tps`      | tok/s  | Prefill 吞吐率                  |
| `decode_tps`       | tok/s  | Decode 吞吐率                   |

`/ocr_layout_image`、`/ocr_layout_pdf` 返回聚合 metrics（所有 block 累加）：

| 指标 | 单位 | 说明 |
|---|---|---|
| `layout_ms` | ms | DocLayout 检测耗时 |
| `layout_boxes` | 个 | 检出 block 数 |
| `vlm_blocks_run` | 个 | 实际跑 VLM 的 block 数 |
| `vlm_blocks_skipped` | 个 | 跳过的 block 数（image 类型） |
| `vision_latency_total_ms` | ms | 所有 block vision 推理总耗时 |
| `prefill_latency_total_ms` | ms | 所有 block prefill 总耗时 |
| `prefill_tokens_total` | 个 | 所有 block prefill token 总数 |
| `decode_latency_total_ms` | ms | 所有 block decode 总耗时 |
| `decode_tokens_total` | 个 | 所有 block decode token 总数 |
| `vlm_total_ms` | ms | VLM 部分总耗时（vision+prefill+decode） |
| `page_total_ms` | ms | 单页总耗时（layout+vlm） |
| `vlm_tps_avg` | tok/s | block 平均 decode TPS |

## 六、注意事项

1. 服务端启动时加载模型需要数秒，请等待日志提示 `Running on ...` 后再发送请求。
2. 推理为单线程串行处理（`threaded=False`），同时只能处理一个请求。
3. 服务端使用 Flask 开发服务器，适合调试和演示；生产环境建议替换为 Gunicorn 等生产级 WSGI 服务器。
4. **NPU 独占使用**：RKNN NPU 一次只能被一个 Python 进程使用。启动新 server 前先 `ps -ef | grep paddleocr_vl_server` 确认无残留；有则 `kill <pid>` 后等 30s+ 让 NPU 内存释放。**绝不能 kill `rknn3_transfer_proxy`**（RK3588↔RK1820 通信代理）；若 NPU 程序异常退出导致内存未释放，直接重启板子。
5. 客户端超时：`--image` / `--layout_image` 600 秒，`--pdf` / `--layout_pdf` 1800 秒。
