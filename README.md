# PaddleOCR-VL Python Binding (pybind11)

RKNN3 版本 PaddleOCR-VL 的 Python 封装，使用 pybind11 将 C++ 推理代码打包为 Python 扩展模块。

## 目录结构
模型文件使用官方python导出模型的方法获得。非本demo重点。

---
1. 本demo，全流程在RK_EVB10_RK3588上完成。如果在pc侧编译，需要调整测试交叉编译脚本并下载相关的gcc、cmake等依赖。
---

## 版面检测模型
PP-DocLayoutV2 onnx模型下载地址：https://huggingface.co/alex-dinh/PP-DocLayoutV2-ONNX/tree/main，目前部署在主控端，占用主控端的资源。
1. PP-DocLayoutV2 onnx 主要用于版面检测。可用于：
    1. 提升当前ocr的识别精度；
    2. 用于排版
2. 存在的问题：
    1. 尝试将PP-DocLayoutV2 onnx模型转为rknn模型部署，但模型导出暂时没有调试通过
    2. 即使导出成功，也缺少libpostprocess_rk182x so包，需要进一步实验是否能调用到rknn3 npu
3. 文件
```
examples/paddleocr_vl/python_binding/
├── PP-DocLayoutV2_onnx # 版面检测模型文件
    ├── export_rknn.py # 导出rknn模型，但导出过程中调试未通过，后面pipeline流程暂且部署在主控端上
    ├── infer_visualize.py  # 本地测试版面检测模型脚本
```


```
examples/paddleocr_vl/
├── cpp/                          # 原有 C++ 推理代码（零改动）
└── python_binding/               # 本项目：pybind11 封装
    ├── CMakeLists.txt            # CMake 构建配置
    ├── pybind_module.cc          # pybind11 绑定层
    ├── build.sh                  # PC 交叉编译脚本（PC → 板端）
    ├── build_local.sh            # 板端本地编译脚本
    ├── demo.py                   # 调用示例
    ├── paddleocr_vl_server.py    # Flask HTTP server（/ocr、/ocr_layout_image、/ocr_layout_pdf）
    ├── paddleocr_vl_client.py    # HTTP client CLI
    ├── run_pipeline.py           # 非HTTP 的 pipeline CLI 入口
    ├── pipeline/                 # DocLayout + VLM 流水线模块
    │   ├── __init__.py           # DEFAULT_PATHS + 25 类 label
    │   ├── doc_layout_runner.py  # PP-DocLayoutV2 ONNX 推理封装
    │   ├── label_mapping.py      # DocLayout label → VLM prompt_type 映射
    │   ├── vlm_runner_so.py      # paddleocr_vl .so 薄包装
    │   ├── vlm_runner.py         # 原 torch 版 runner（保留，未被 pipeline 使用）
    │   ├── pipeline.py           # 流水线编排：layout → crop → VLM → 聚合
    │   ├── metrics_aggregator.py # 单页 metrics 聚合
    │   ├── pages_parser.py       # PDF pages 参数解析
    │   └── result.py             # PipelineResult / PaddleOCRVLBlock
    ├── tests/                    # pytest 单元/集成测试
    ├── COMPILE_PLANS.md          # 编译方案说明
    └── BUILD_AND_TEST.md         # 编译部署文档
    └── README_CS_Demo.md         # C/S 推理Demo文档
    └── paddleocr_vl_client.py    # C/S 推理客户端
    └── paddleocr_vl_server.py    # C/S 推理服务端
    └── README.md                 # 本文件
```

## 安装依赖

```bash
# PC 端交叉编译需要：
# 1. pybind11
pip install pybind11

# 2. 交叉编译工具链
export GCC_COMPILER=/home/nianliu/tmp/rknn3/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

# 板端本地编译需要：
# apt-get install python3.11-dev cmake gcc g++ make
```

## 编译

### 方案 1：板端本地编译（推荐，直接运行）

板端需要 Python 3.11-dev 已安装（Python.h + libpython3.11.so）。

```bash
cd examples/paddleocr_vl/python_binding/
./build_local.sh -t rk3588 -b Release
```

输出在 `build/install_rk3588_Release/paddleocr_vl/`。

### 方案 2：PC 交叉编译

PC 上交叉编译输出 aarch64 的 `.so`，然后 scp 到板端。

```bash
cd examples/paddleocr_vl/python_binding/
./build.sh -t rk3588 -a aarch64 -b Release
```

输出在 `build/install_rk3588_aarch64/paddleocr_vl/`。

> **注意**：PC 端交叉编译需要指定正确的 Python 版本头文件和库。
> 详见 [COMPILE_PLANS.md](./python_binding/COMPILE_PLANS.md)。

### 验证

```bash
# 交叉编译产物确认
$ file build/install_rk3588_aarch64/paddleocr_vl/*.so
# → ELF 64-bit LSB shared object, ARM aarch64 ✓

# 板端确认
$ file /usr/lib/python3/dist-packages/paddleocr_vl/*.so
# → ARM aarch64 ✓
```

## 部署到板端

```bash
# 1. 复制 .so 到板端 Python site-packages
scp build/install_rk3588_aarch64/paddleocr_vl/*.so root@board_ip:/usr/lib/python3/dist-packages/

# 2. 确保板端模型文件已部署（同现有 cpp demo 的 model 目录）
scp -r model/ root@board_ip:/path/to/model/

# 3. 运行示例
python3 /path/on/board/demo.py
```

> **Python 版本必须匹配**：编译出的 `.so` 文件名包含 Python 版本（如 `python311`），
> 必须与板端 Python 版本一致。PC 端 Python 3.13 编译的版本不能在板端 Python 3.11 上使用。

## Python API

### 基本用法

```python
import paddleocr_vl

# 创建模型实例
model = paddleocr_vl.PaddleOCRVL()

# 初始化：加载 vision + mlpar + llm 三个子模型
model.init(
    model_dir="model",              # 模型目录
    vision_core_mask=0x1,           # vision 使用的 NPU core mask
    mlpar_core_mask=0x2,            # MLP-AR 使用的 NPU core mask
    llm_core_mask=0x4,              # LLM 使用的 NPU core mask
    model_width=504,                # 模型输入宽度（默认 504）
    model_height=504,               # 模型输入高度（默认 504）
)

# 推理
result = model.run(
    image_path="data/vision/test.png",
    prompt="OCR:",                  # 提示词
    prompt_type="ocr",              # 提示类型: "ocr"/"table"/"chart"/"formula"/"image"
)

# 结果
print(result["text"])
# → "这是一张包含表格的图片..."
print(result["metrics"])
# → {
#     "vision_latency": 123.45,     # 毫秒
#     "prefill_latency": 45.67,
#     "decode_latency": 89.01,
#     "ttft": 12.3,                 # 首 token 时间
#     "prefill_tokens": 100,
#     "decode_tokens": 50,
#     "tpt": 1.2,                   # 每 token 时间
#     "tps": 50.0,                  # 每秒 token 数
# }

# 释放资源
model.release()
```

### Prompt 类型

| `prompt_type` | 实际 prompt |
|---|---|
| `ocr` | `OCR:` |
| `table` | `Table Recognition:` |
| `chart` | `Chart Recognition:` |
| `formula` | `Formula Recognition:` |
| `image` | `Image Recognition:` |

自定义 prompt 可直接传入 `prompt` 参数。

## 模型文件路径

`init()` 会自动从 `model_dir` 查找以下文件：

```
model/
├── PaddleOCR-vision.rknn           # vision 子模型
├── PaddleOCR-vision.weight
├── PaddleOCR-vision-mlp_AR.rknn    # MLP-AR 子模型
├── PaddleOCR-vision-mlp_AR.weight
├── PaddleOCR-llm.rknn              # LLM 子模型
├── PaddleOCR-llm.weight
└── llm/
    ├── PaddleOCR-llm.tokenizer.gguf
    └── PaddleOCR-llm.embed.bin
└── vision/
    └── position_embedding_model.bin
```

## 核心设计

### 初始化流程

1. 加载 tokenizer（`Tokenizer` 库）
2. mmap 加载 embedding 文件（`embed.bin`）
3. 配置 LLM 参数（sampling 参数、vocab_info）
4. 注册自定义 callback（收集 LLM 输出文本，而非 printf）
5. 初始化 vision/llm/mlpar 三个子模型
6. 分配 vision_embeds 和 img_embeds 内存

### 推理流程

1. 读取图片 → `image_buffer_t`
2. 构建 prompt（自动添加 `<image>` 标签）
3. 构建 `rknn3_llm_multimodal_tensor`
4. 调用 `inference_paddleocr_vl_model()`
5. 收集 result_callback 输出的文本
6. 计算性能指标

### Callback 定制

C++ 原版 `result_callback` 硬编码 `printf`，绑定层提供了自定义 callback：

- `py_result_callback`：将 token 累积到 `std::string`，线程安全（mutex）
- `py_tokenizer_callback`：复用外部 Tokenizer 实例
- `py_embed_callback`：复用 mmap 的 embedding 数据

## 与 Flask 集成

`paddleocr_vl_server.py` 提供以下路由：

| 路由 | 用途 |
|---|---|
| `/ocr` | 单图直跑 VLM |
| `/ocr_base64` | base64 JSON 单图直跑 |
| `/ocr_pdf` | PDF 逐页直跑 |
| `/ocr_layout_image` | 单图 DocLayout → 切 block → 逐 block VLM |
| `/ocr_layout_pdf` | PDF 逐页 Layout OCR |
| `/health` | 健康检查 |

**接口详细说明、请求/响应字段、curl 示例、客户端完整参数和输出文件保存规则**见 [README_CS_Demo.md](./python_binding/README_CS_Demo.md)。

### 单进程单实例约束（重要）

`paddleocr_vl.PaddleOCRVL()` 在同一 Python 进程内**不能创建两个实例**。`.so` 内部全局状态（KV cache、tokenizer、callback context、internal share memory）在第二个 `init()` 时会被覆盖，导致两个实例的 `run()` 都返回空文本（NPU 推理照常执行，但 callback 收不到内容）。

**因此 server.py 强制约定**：

- `init_model()` 创建一个 `PaddleOCRVL` 单例，给 `/ocr` 路由使用
- `init_pipeline()` **不再新建** `PaddleOCRVL` 实例，而是通过 `Pipeline(model=model)` 把 `/ocr` 路由的实例传给 `VLMRunnerSo`
- `VLMRunnerSo(model=...)` 接受外部实例时跳过 `init()` 和 `release()`，只复用
- pipeline 的 `close()` 也不会 release 共享实例（由 `/ocr` 路由的全局 `model` 负责）

如果同时需要 `/ocr` 和 `/ocr_layout_*` 路由共存，**必须**采用这种共享实例模式，不能各建各的。

### 板端运行

板端前置条件（已就位）：
- `.so` 在 `build/build_local_rk3588_Release/paddleocr_vl.cpython-311-aarch64-linux-gnu.so`
- 模型在 `/root/demo_paddleocr_vl/model`
- DocLayout ONNX 在 `/root/demo_paddleocr_vl/PP_onnx/PP-DocLayoutV2.onnx`

```bash
# 板端启动 server
source /userdata/venv/bin/activate
cd /root/demo_paddleocr_vl/rknn3-model-zoo/examples/paddleocr_vl/python_binding
python paddleocr_vl_server.py --port 8080

# 客户端调用（任意机器），结果保存到 ./output
python paddleocr_vl_client.py --server http://localhost:8080 \
    --layout_image /path/to/page.png --output_dir ./output --visualize

python paddleocr_vl_client.py --server http://localhost:8080 \
    --layout_pdf /path/to/doc.pdf --pages "1-3,5" --output_dir ./output
```

### NPU 独占使用

RKNN NPU 在板端是**独占资源**，一次只能有一个 Python 进程使用：

- 启动新 server 前，先 `ps -ef | grep paddleocr_vl_server` 确认无残留；有则 `kill <pid>` 后等 30s+ 让 NPU 内存释放
- **绝不能 kill `rknn3_transfer_proxy`**（RK3588↔RK1820 通信代理）；杀掉它 NPU 内存也不释放，只能重启板子
- 若 NPU 程序异常退出导致 `rknn-smi info -w` 显示内存占用未清零，直接重启板子，不要试图清理

### 小 block padding

DocLayout 检出的 `text` 类 bbox 通常很扁（如 1004×57），直接送 `.so` 会被内部 resize 到极端比例（如 2268×112），影响识别稳定性。`pipeline.py` 中的 `_pad_to_min()` 会把短边 < `MIN_BLOCK_SIZE` (200) 的 crop 用白底 padding 到 ≥200，保留更接近训练分布的长宽比。已验证 padding 不改变 `.so` 输出内容，仅提升对小 block 的鲁棒性。

## 非HTTP CLI

`run_pipeline.py` 提供命令行入口，不走 HTTP，直接跑 pipeline：

```bash
# 单图
python run_pipeline.py --image /path/to/page.png --output_dir ./output

# PDF
python run_pipeline.py --pdf /path/to/doc.pdf --output_dir ./output
```

输出：`result_<stem>.json` / `.md` / `.html` / `_vis.png`。

## 测试

```bash
# 板端或 PC（PC 上不能 import 真实 paddleocr_vl .so，但所有测试都用 mock）
conda activate manba_env   # 或对应环境
python -m pytest tests/ -q
```

测试覆盖：
- `test_label_mapping.py` — 25 类 label → prompt_type 映射
- `test_result.py` — `to_markdown` / `to_html` / `save_all`
- `test_metrics_aggregator.py` — 单页 metrics 累加 + 派生字段
- `test_pages_parser.py` — PDF `pages` 参数解析（合法 + 非法）
- `test_vlm_runner_so.py` — `VLMRunnerSo` 路径/PIL/ndarray 输入、tmp 清理、外部实例复用
- `test_pipeline_e2e.py` — mock DocLayout + VLM，端到端编排

