#!/bin/bash
# =============================================================================
# 方案 1：在板端本地编译 Python 3.11
# 适用于板端有完整编译环境（gcc + cmake + python3-dev）
# =============================================================================
set -e

TARGET_SOC=""
BUILD_TYPE="Release"
PY_VERSION="3.11"

echo "$0 $@"
while getopts ":t:b:" opt; do
  case $opt in
    t) TARGET_SOC=$OPTARG ;;
    b) BUILD_TYPE=$OPTARG ;;
    :)
      echo "Option -$OPTARG requires an argument."
      exit 1
      ;;
    ?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

if [ -z ${TARGET_SOC} ]; then
  echo "$0 -t <target> [-b <build_type>]"
  echo ""
  echo "    -t : target (rk3588/rk3576)"
  echo "    -b : build_type (Debug/Release, default: Release)"
  exit -1
fi

# Use local compiler
export CC=gcc
export CXX=g++

# Detect Python 3.11
if command -v python3.11 >/dev/null 2>&1; then
    PY_EXECUTABLE=$(which python3.11)
elif python3 -c "import sys; exit(0 if sys.version_info[:2] >= (3,11) else 1)" 2>/dev/null; then
    PY_EXECUTABLE=$(which python3)
else
    echo "Python 3.11 not found on this system."
    echo "Please install: apt-get install python3.11 python3.11-dev"
    exit 1
fi

PY_INCLUDE=$($PY_EXECUTABLE -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_LIBDIR=$($PY_EXECUTABLE -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "Python: ${PY_EXECUTABLE} (${PY_VERSION})"
echo "Python Include: ${PY_INCLUDE}"
echo "Python Libdir: ${PY_LIBDIR}"
echo "==================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/build_local_${TARGET_SOC}_${BUILD_TYPE}"
INSTALL_DIR="${SCRIPT_DIR}/build/install_${TARGET_SOC}_${BUILD_TYPE}"

mkdir -p ${BUILD_DIR}
rm -rf ${INSTALL_DIR}
cd ${BUILD_DIR}

cmake ${SCRIPT_DIR} \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_ASAN=OFF \
    -DDISABLE_RGA=ON \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DPython3_EXECUTABLE=${PY_EXECUTABLE} \
    -DPython3_INCLUDE_DIR=${PY_INCLUDE} \
    -DPython3_LIBRARY=${PY_LIBDIR}/libpython3.11.so

make -j$(nproc)

# Find the .so file
SO_FILE=$(find ${BUILD_DIR} -name "paddleocr_vl.cpython-*.so")
if [ -n "${SO_FILE}" ]; then
    mkdir -p ${INSTALL_DIR}/paddleocr_vl
    cp ${SO_FILE} ${INSTALL_DIR}/paddleocr_vl/
    echo ""
    echo "=================================================="
    echo "Built successfully: ${INSTALL_DIR}/paddleocr_vl/"
    echo "=================================================="
    echo ""
    echo "To install on board:"
    echo "  1. scp ${INSTALL_DIR}/paddleocr_vl/*.so root@board_ip:/usr/lib/python3/dist-packages/"
    echo "  2. On board: python3 demo.py"
else
    echo "ERROR: .so file not found in build directory"
    exit 1
fi
