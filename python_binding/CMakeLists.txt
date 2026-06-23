cmake_minimum_required(VERSION 3.10)

project(rknn_paddleocr_vl_pybind)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# ── Demo root (../ directory) ──────────────────────────────────
set(DEMO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ── 3rdparty and utils (relative to demo root: ../../) ─────────
set(THIRDPARTY_ROOT ${DEMO_ROOT}/../../3rdparty/)
set(UTILS_ROOT ${DEMO_ROOT}/../../utils/)

add_subdirectory(${THIRDPARTY_ROOT} 3rdparty.out)
add_subdirectory(${UTILS_ROOT} utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN")

# ── Source files (copy from cpp/ excluding main.cc) ────────────
file(GLOB CPP_SRCS ${DEMO_ROOT}/cpp/*.cc)
file(GLOB VISION_SRCS ${DEMO_ROOT}/cpp/vision/*.cc)
file(GLOB LLM_SRCS ${DEMO_ROOT}/cpp/llm/*.cc)
file(GLOB MLPAR_SRCS ${DEMO_ROOT}/cpp/mlpar/*.cc)

# Remove main.cc from the list (it's just the demo entry point)
foreach(src ${CPP_SRCS})
    get_filename_component(fname "${src}" NAME)
    if(fname STREQUAL "main.cc" OR fname STREQUAL "vlm.cc")
        list(REMOVE_ITEM CPP_SRCS "${src}")
    endif()
endforeach()

# ── Pybind11 module ────────────────────────────────────────────
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        RESULT_VARIABLE _pip_result
        OUTPUT_VARIABLE _pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _pip_result)
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
        find_package(pybind11 REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found for ${Python3_EXECUTABLE}. Install with: pip install pybind11")
    endif()
endif()

pybind11_add_module(paddleocr_vl
    ${CPP_SRCS}
    ${VISION_SRCS}
    ${LLM_SRCS}
    ${MLPAR_SRCS}
    pybind_module.cc
)

target_link_libraries(paddleocr_vl
    PRIVATE
    timeutils
    imageutils
    fileutils
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(paddleocr_vl
    PRIVATE
        pthread
    )
endif()

# Include paths: original cpp/ dir has #include "llm/..." etc.
target_include_directories(paddleocr_vl PRIVATE
    ${DEMO_ROOT}/cpp
    ${DEMO_ROOT}/cpp/vision
    ${DEMO_ROOT}/cpp/llm
    ${DEMO_ROOT}/cpp/mlpar
    ${LIBRKNNAPI_INCLUDES}
    ${LIBTOKENIZER_INCLUDES}
)

# ── Install ─────────────────────────────────────────────────────
install(TARGETS paddleocr_vl
    DESTINATION paddleocr_vl
)

# Also install model files
file(GLOB MODEL_RKNN "${DEMO_ROOT}/model/*/*.rknn")
file(GLOB MODEL_WEIGHT "${DEMO_ROOT}/model/*/*.weight")
file(GLOB MODEL_TOKEN "${DEMO_ROOT}/model/llm/*.tokenizer.gguf")
file(GLOB MODEL_EMBED "${DEMO_ROOT}/model/llm/*.embed.bin")
file(GLOB MODEL_POS_EMBED "${DEMO_ROOT}/model/vision/position_embedding_model.bin")
install(FILES ${MODEL_RKNN} DESTINATION model)
install(FILES ${MODEL_WEIGHT} DESTINATION model)
install(FILES ${MODEL_TOKEN} DESTINATION model)
install(FILES ${MODEL_EMBED} DESTINATION model)
install(FILES ${MODEL_POS_EMBED} DESTINATION model)
