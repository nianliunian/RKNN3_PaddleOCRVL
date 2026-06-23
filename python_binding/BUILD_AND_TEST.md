# PaddleOCR-VL Python Binding 编译部署文档

## 1. 项目概述

将 PaddleOCR-VL 的 C++ 推理代码通过 pybind11 封装为 Python 模块 `paddleocr_vl`，可在 RK3588 板端直接通过 Python 调用模型进行 OCR 推理。

### 目录结构
需要先拉取rknn3-model-zoo的仓库https://github.com/airockchip/rknn3-model-zoo，然后将`python_binding`放置到如下目录所示。

---
- 本demo，全流程在RK_EVB10_RK3588上完成，理论上在PC侧交叉编译后将包push到板端也可以
- 本demo，全流程在RK_EVB10_RK3588上完成，理论上在PC侧交叉编译后将包push到板端也可以
- 本demo，全流程在RK_EVB10_RK3588上完成，理论上在PC侧交叉编译后将包push到板端也可以
---

```
paddleocr_vl/
├── cpp/                        # C++ 原始代码
│   ├── main.cc                 # 命令行 demo 入口
│   ├── vlm.cc                  # VLM 封装（C 接口）
│   ├── paddleocr_vl.cc/h       # 核心推理逻辑
│   ├── llm/                    # LLM 子模型
│   ├── vision/                 # Vision 子模型
│   └── mlpar/                  # MLP-AR 子模型
├── python_binding/             # pybind11 绑定
│   ├── CMakeLists.txt
│   ├── pybind_module.cc        # pybind11 绑定代码
│   ├── build_local.sh          # 本地编译脚本
│   ├── build.sh                # 交叉编译脚本
│   └── test_paddleocr_vl_lib.py  # 测试脚本
├── data/                       # 测试图片
└── model/                      # 模型文件
```

## 2. 编译

### 2.1 环境要求

| 依赖         | 版本要求       | 说明                              |
|-------------|---------------|-----------------------------------|
| GCC         | 12.2+         | 板端自带即可                       |
| CMake       | 3.10+         | 板端自带即可                       |
| Python      | 3.11          | python3.11-dev 需安装              |
| pybind11    | 2.x           | pip install pybind11               |

### 2.2 检查依赖

```bash
# 检查 Python 版本
python3 --version    # 需 3.11，可以使用venv的虚拟环境

# 检查 pybind11
python3 -c "import pybind11; print(pybind11.get_cmake_dir())"

# 检查编译器
gcc --version
cmake --version
```

### 2.3 编译步骤

```bash
cd /root/demo_paddleocr_vl/rknn3-model-zoo/examples/paddleocr_vl/python_binding

# 创建构建目录
mkdir -p build/build_local_rk3588_Release

cd build/build_local_rk3588_Release

# CMake 配置
cmake /root/demo_paddleocr_vl/rknn3-model-zoo/examples/paddleocr_vl/python_binding \
    -DTARGET_SOC=rk3588 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_ASAN=OFF \
    -DDISABLE_RGA=ON \
    -DCMAKE_INSTALL_PREFIX=../install_rk3588_Release

# 编译
make -j$(nproc)
```

### 2.4 编译产物

编译成功后，.so 文件位于：

```
build/build_local_rk3588_Release/paddleocr_vl.cpython-311-aarch64-linux-gnu.so
```

也可以使用 build_local.sh 一键编译：

```bash
bash build_local.sh -t rk3588 -b Release
```

## 3. 部署

### 3.1 模型文件准备

模型文件需放在统一目录，默认路径 /root/demo_paddleocr_vl/model/，结构如下：

```
model/
├── PaddleOCR-vision.rknn
├── PaddleOCR-vision.weight
├── PaddleOCR-llm.rknn
├── PaddleOCR-llm.weight
├── PaddleOCR-vision-mlp_AR.rknn
├── PaddleOCR-vision-mlp_AR.weight
├── vision/
│   └── position_embedding_model.bin
└── llm/
    ├── PaddleOCR-llm.tokenizer.gguf
    └── PaddleOCR-llm.embed.bin
```

### 3.2 rknn3_transfer 代理

RK3588 与 RK1820 之间的通信依赖 rknn3_transfer 进程，**切勿手动 kill**，否则 NPU 推理将失败。

```bash
# 检查进程是否运行
ps aux | grep rknn3_transfer | grep -v grep

# 正常应看到类似输出：
# root  1668  /bin/rknn3_transfer_proxy
# root  4468  rknn3_transfer_proxy_237c40f1 -s 0003:31:00.0
```

## 4. 测试

### 4.1 测试脚本

```python
import sys
lib_dir = r"/root/demo_paddleocr_vl/rknn3-model-zoo/examples/paddleocr_vl/python_binding/build/build_local_rk3588_Release"
sys.path.insert(0, lib_dir)
import paddleocr_vl

model = paddleocr_vl.PaddleOCRVL()
model.init(r"/root/demo_paddleocr_vl/model", vision_core_mask=0xff, mlpar_core_mask=0xff, llm_core_mask=0xff)
result = model.run(r"/root/demo_paddleocr_vl/model/test.png")

print()
print("=" * 90)
print("OCR Result:")
print(result["text"])
print()

m = result["metrics"]
print("-" * 90)
hdr = "  {:<12} {:>15} {:>8} {:>18} {:>18}".format(
    "Stage", "Total Time (ms)", "Tokens", "Time/Token (ms)", "Tokens/sec")
print(hdr)
print("-" * 90)

prefill_ms = m["prefill_latency"]
prefill_tokens = int(m["prefill_tokens"])
prefill_tpt = m["prefill_tpt"]
prefill_tps = m["prefill_tps"]
print("  {:<12} {:>15.2f} {:>8} {:>18.2f} {:>18.2f}".format(
    "Prefill", prefill_ms, prefill_tokens, prefill_tpt, prefill_tps))

decode_ms = m["decode_latency"]
decode_tokens = int(m["decode_tokens"])
decode_tpt = m["decode_tpt"]
decode_tps = m["decode_tps"]
print("  {:<12} {:>15.2f} {:>8} {:>18.2f} {:>18.2f}".format(
    "Generate", decode_ms, decode_tokens, decode_tpt, decode_tps))

print("-" * 90)
print("  TTFT: {:.2f} ms".format(m["ttft"]))
print("  Vision latency: {:.2f} ms".format(m["vision_latency"]))
print("  Total tokens: {}".format(prefill_tokens + decode_tokens))
print("  Overall TPS: {:.2f} tokens/s".format(m["tps"]))
print("=" * 90)

model.release()
```

### 4.2 运行测试

```bash
cd /root/demo_paddleocr_vl/rknn3-model-zoo/examples/paddleocr_vl/python_binding
python3 test_paddleocr_vl_lib.py
```

### 4.3 预期输出

```
==========================================================================================
OCR Result:
考评项目 | 权重 | 考评内容 | 评分标准 | 评分方法 | 自评得分
一级领导体系 (120分) | **学校安全 工作目标 责任制** | 20 | ...

------------------------------------------------------------------------------------------
  Stage        Total Time (ms)   Tokens    Time/Token (ms)         Tokens/sec
------------------------------------------------------------------------------------------
  Prefill                59.06      338               0.17            5722.80
  Generate             1565.13      368               4.25             235.12
------------------------------------------------------------------------------------------
  TTFT: 59.06 ms
  Vision latency: 529.97 ms
  Total tokens: 706
  Overall TPS: 0.43 tokens/s
==========================================================================================
```

### 4.4 API 说明

```python
import paddleocr_vl

# 创建实例
model = paddleocr_vl.PaddleOCRVL()

# 初始化模型
model.init(
    model_dir,              # 模型文件根目录，默认 "model"
    vision_core_mask=0x1,   # Vision 子模型 NPU 核掩码
    mlpar_core_mask=0x2,    # MLP-AR 子模型 NPU 核掩码
    llm_core_mask=0x4,      # LLM 子模型 NPU 核掩码
    model_width=504,        # 模型输入宽度
    model_height=504        # 模型输入高度
)

# 推理
result = model.run(
    image_path,             # 图片路径
    prompt="OCR:",          # 提示词
    prompt_type="ocr"       # 提示词类型: ocr/table/chart/formula/image
)

# 返回结果
# result["text"]     - OCR 识别文本
# result["metrics"]  - 性能数据 dict:
#   vision_latency   - Vision 推理延迟 (ms)
#   prefill_latency  - Prefill 阶段延迟 (ms)
#   decode_latency   - Decode 阶段延迟 (ms)
#   ttft             - 首Token延迟 (ms)
#   prefill_tokens   - Prefill token 数
#   decode_tokens    - Decode token 数
#   tpt              - 每Token平均延迟 (ms)
#   tps              - 每秒Token数
#   prefill_tpt/tps  - Prefill 阶段性能
#   decode_tpt/tps   - Decode 阶段性能

# 释放资源
model.release()
```

## 5. 遇到的问题及解决方案

### 问题 1：ModuleNotFoundError: No module named paddleocr_vl

**现象**：Python 导入 paddleocr_vl 时报 ModuleNotFoundError。

**原因**：测试脚本 test_paddleocr_vl_lib.py 中引用的 .so 路径是旧路径 `build/install_rk3588_Release/...`，而实际编译产物在 `build/build_local_rk3588_Release/` 目录下。`install_rk3588_Release` 目录从未生成（未执行 `make install`）。

**解决**：修正 `sys.path.insert` 中的路径为实际 .so 所在目录：

```python
lib_dir = r"/root/demo_paddleocr_vl/rknn3-model-zoo/examples/paddleocr_vl/python_binding/build/build_local_rk3588_Release"
```

### 问题 2：性能数据异常（Prefill 延迟达到万亿毫秒）

**现象**：输出的性能数据中 Prefill 和 Generate 的延迟值异常巨大（约 1.78 万亿 ms），TPS 为 0 或 inf。

**原因**：`rknn_paddleocr_vl_llm.cc` 中计算时间戳的两行代码被注释掉了：

```c
// perf->llm_start_time = getCurrentTimeUs();
ret = rknn3_session_run(llm_ctx->rknn_sess, inputs, n_inputs, &llm_infer_param);
// perf->llm_end_time = getCurrentTimeUs();
```

`perf` 在调用方通过 `memset` 初始化为 0，但 `rknn_perf_metrics_t` 的 `llm_start_time` 和 `llm_end_time` 字段没有被正确赋值。而回调中记录的 `first_token` 时间戳是正常值，导致 `first_token - perf->llm_start_time` 产生巨大的差值。

**解决**：取消注释这两行代码，重新编译：

```c
perf->llm_start_time = getCurrentTimeUs();
ret = rknn3_session_run(llm_ctx->rknn_sess, inputs, n_inputs, &llm_infer_param);
perf->llm_end_time = getCurrentTimeUs();
```

修改文件：`cpp/llm/rknn_paddleocr_vl_llm.cc` 第 127、129 行。

### 问题 3：SSH 长时间推理连接断开

**现象**：通过 SSH 远程运行推理时，连接在推理过程中被断开（Connection to 192.168.1.102 closed by remote host）。

**原因**：LLM 推理耗时较长（约 1.5s），期间无输出导致 SSH 空闲超时。

**解决**：SSH 连接时添加 keepalive 参数：

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=20 root@192.168.1.102 "python3 test_paddleocr_vl_lib.py"
```

### 问题 4：pybind_module.cc 中符号重复定义

**说明**：`vlm.cc` 和 `main.cc` 中定义了全局变量（如 `SAMPLE_PARAMS`、`prompt_prefix`、`prompt_postfix` 等）和回调函数（`result_callback`、`tokenizer_callback`、`embed_callback`），与 `pybind_module.cc` 中的定义冲突。

**解决**：CMakeLists.txt 中已通过 `foreach` 排除了 `main.cc` 和 `vlm.cc`，`pybind_module.cc` 重新定义了这些全局变量和回调，确保符号唯一。核心推理逻辑（`paddleocr_vl.cc`、`llm/`、`vision/`、`mlpar/`）被共享链接。
