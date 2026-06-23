# PaddleOCR-VL 服务端推理 Demo (Client-Server 模式)

基于 RKNN3 的 PaddleOCR-VL C/S 架构推理示例。服务端部署在 RK3588 开发板上，客户端通过 HTTP 接口发送图片，获取 OCR 识别结果。

## 文件说明

```
python_binding/
├── build/build_local_rk3588_Release/
│   └── paddleocr_vl.cpython-311-aarch64-linux-gnu.so  # pybind11 编译的推理库
├── paddleocr_vl_server.py   # 服务端 — 加载模型，提供 HTTP API
├── paddleocr_vl_client.py   # 客户端 — 发送图片，显示 OCR 结果
├── test_paddleocr_vl_lib.py # 本地单图推理测试
└── demo.py                  # 本地多模式推理 demo
```

模型文件目录：`/root/demo_paddleocr_vl/model/`

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

### 1. `POST /ocr` — 表单上传图片

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

### 3. `GET /health` — 健康检查

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

### 基本用法

```bash
# OCR 文字识别（默认）
python3 paddleocr_vl_client.py --image /root/demo_paddleocr_vl/model/test.png

# 指定服务端地址
python3 paddleocr_vl_client.py --server http://192.168.46.102:8080 --image test.png
```

### 完整参数

| 参数           | 默认值                           | 说明                     |
|---------------|----------------------------------|--------------------------|
| `--server`    | `http://192.168.46.102:8080`     | 服务端地址                |
| `--image`     | （必填）                         | 图片文件路径              |
| `--prompt`    | `OCR:`                           | 自定义 prompt 文本        |
| `--prompt_type` | `ocr`                          | 预设 prompt 类型           |

### 不同模式示例

```bash
# 文字识别
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

返回的 `metrics` 字段含义：

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

## 六、注意事项

1. 服务端启动时加载模型需要数秒，请等待日志提示 `Running on ...` 后再发送请求。
2. 推理为单线程串行处理（`threaded=False`），同时只能处理一个请求。
3. 服务端使用 Flask 开发服务器，适合调试和演示；生产环境建议替换为 Gunicorn 等生产级 WSGI 服务器。
4. 客户端超时设置为 120 秒，复杂图片推理可能需要较长时间。
