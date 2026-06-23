# PaddleOCR-VL Python 绑定编译方案

## 方案 1：板端本地编译

### 适用场景
- 板端有完整的编译工具链（gcc, cmake, make）
- 板端 Python 3.11 有 dev 包（`/usr/include/python3.11/Python.h`）

### 前置条件

```bash
# 板端确认
python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
# 应返回: /usr/include/python3.11

which gcc && which cmake && which make
```

如果缺少 Python.h，先安装：
```bash
# Debian/Ubuntu
apt-get install python3.11-dev

# 或使用 apt
apt-get install python3-dev
```

### 编译步骤

```bash
# 1. 部署 .so 到板端
cd /path/to/rknn3-model-zoo/examples/paddleocr_vl/python_binding
cp demo.py /path/on/board/

# 2. 在板端编译
mkdir build && cd build
cmake .. \
    -DTARGET_SOC=rk3588 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_INSTALL_PREFIX=. \
    -DPython3_EXECUTABLE=$(which python3) \
    -DPython3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
    -DPython3_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so

make -j4

# 3. 复制 .so 到 site-packages
mkdir -p /usr/lib/python3/dist-packages/paddleocr_vl
cp paddleocr_vl.cpython-311-aarch64-linux-gnu.so /usr/lib/python3/dist-packages/paddleocr_vl/

# 4. 测试
cd /path/on/board
python3 demo.py
```

### 优点
- 编译环境完全匹配
- 无需交叉编译工具链
- 可直接使用 `python3 -m pip install pybind11`

### 缺点
- 板端资源有限，编译较慢
- 需要安装额外的 dev 包

---

## 方案 2：PC 交叉编译 Python 3.11

### 适用场景
- PC 有交叉编译工具链
- 需要板端无完整编译环境
- 希望 CI/CD 流水线化

### 前置条件

```bash
# PC 端需要：
# 1. 交叉编译工具链（已提供）
export GCC_COMPILER=/home/nianliu/tmp/rknn3/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

# 2. Python 3.11 交叉编译头文件和库
# 方式 A: 从板端 copy
scp root@board:/usr/include/python3.11/Python.h ./python311_include/
scp root@board:/usr/lib/libpython3.11.so ./python311_lib/

# 方式 B: 在 PC 上安装 aarch64 版 Python 3.11
# apt-get install python3.11-dev-aarch64-cross  (如果可用)

# 3. pybind11
pip install pybind11
```

### 编译步骤

```bash
cd examples/paddleocr_vl/python_binding

# 使用 python311 的 build 脚本
./build_python311.sh -t rk3588 -a aarch64 \
    -DINCLUDE=/path/to/python311/include \
    -DLIB=/path/to/python311/lib

# 输出在 build/install_rk3588_aarch64/paddleocr_vl/
```

### CMake 配置要点

```cmake
# 指定 Python 3.11
set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)
```

### 优点
- PC 编译速度快（多核）
- 无需板端编译环境
- 可 CI/CD 自动化

### 缺点
- 需要 Python 3.11 的头文件和库
- 交叉编译可能遇到链接问题
- 需要额外维护 Python 版本配置

---

## 推荐选择

| 场景 | 推荐方案 |
|------|---------|
| 开发调试 | 方案 1（板端本地编译） |
| 生产部署 | 方案 2（PC 交叉编译，保证一致） |
| 快速验证 | 方案 1（最简单） |
