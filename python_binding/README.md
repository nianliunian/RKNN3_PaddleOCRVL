# PaddleOCR-VL Python Binding (pybind11)

RKNN3 版本 PaddleOCR-VL 的 Python 封装，使用 pybind11 将 C++ 推理代码打包为 Python 扩展模块。

## 目录结构

```
examples/paddleocr_vl/
├── cpp/                          # 原有 C++ 推理代码（零改动）
└── python_binding/               # 本项目：pybind11 封装
    ├── CMakeLists.txt            # CMake 构建配置
    ├── pybind_module.cc          # pybind11 绑定层
    ├── build.sh                  # PC 交叉编译脚本（PC → 板端）
    ├── build_local.sh            # 板端本地编译脚本
    ├── demo.py                   # 调用示例
    ├── COMPILE_PLANS.md          # 编译方案说明
    └── BUILD_AND_TEST.md         # 编译部署文档
    └── README_CS_Demo.md         # C/S 推理Demo文档
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
> 详见 [COMPILE_PLANS.md](COMPILE_PLANS.md)。

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

```python
from flask import Flask, request
import paddleocr_vl

app = Flask(__name__)
model = paddleocr_vl.PaddleOCRVL()
model.init(model_dir="/path/to/model/")

@app.route('/ocr', methods=['POST'])
def ocr():
    image = request.files['image']
    prompt = request.form.get('prompt', 'OCR:')
    result = model.run(image.filename, prompt, request.form.get('prompt_type', 'ocr'))
    return {"text": result["text"], "metrics": result["metrics"]}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```
