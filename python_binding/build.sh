#!/bin/bash
set -e

TARGET_SOC=""
TARGET_ARCH=""
BUILD_TYPE="Release"

echo "$0 $@"
while getopts ":t:a:b:" opt; do
  case $opt in
    t) TARGET_SOC=$OPTARG ;;
    a) TARGET_ARCH=$OPTARG ;;
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

if [ -z ${TARGET_SOC} ] || [ -z ${TARGET_ARCH} ]; then
  echo "$0 -t <target> -a <arch> [-b <build_type>]"
  echo ""
  echo "    -t : target (rk3588/rk3576)"
  echo "    -a : arch (aarch64/arm64)"
  echo "    -b : build_type (Debug/Release, default: Release)"
  exit -1
fi

# Set cross compiler
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

if ! command -v ${CC} >/dev/null 2>&1; then
    echo "${CC} is not available"
    echo "Please set GCC_COMPILER env var"
    echo "such as: export GCC_COMPILER=/home/nianliu/tmp/rknn3/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu"
    exit 1
fi

echo "Using cross compiler: $CC ($(file $(which ${CC})))"
echo "Using cross compiler: $CXX ($(file $(which ${CXX})))"

# Disable RGA (not needed for python binding)
export DISABLE_RGA=ON

# Python config
PYBIND3_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "PYTHON_INCLUDE=${PYBIND3_INCLUDE}"
echo "==================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/build_pybind_${TARGET_SOC}_${TARGET_ARCH}_${BUILD_TYPE}"
INSTALL_DIR="${SCRIPT_DIR}/build/install_${TARGET_SOC}_${TARGET_ARCH}"

mkdir -p ${BUILD_DIR}
rm -rf ${INSTALL_DIR}
cd ${BUILD_DIR}

# Pass compiler paths explicitly to cmake to avoid using host compiler
cmake ${SCRIPT_DIR} \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_ASAN=OFF \
    -DDISABLE_RGA=${DISABLE_RGA} \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DPython3_EXECUTABLE=$(which python3)

make -j$(nproc)

# Find the .so file
SO_FILE=$(find ${BUILD_DIR} -name "paddleocr_vl.cpython-*.so")
if [ -n "${SO_FILE}" ]; then
    mkdir -p ${INSTALL_DIR}/paddleocr_vl
    cp ${SO_FILE} ${INSTALL_DIR}/paddleocr_vl/
    echo "Built: ${INSTALL_DIR}/paddleocr_vl/"
else
    echo "ERROR: .so file not found in build directory"
    exit 1
fi
