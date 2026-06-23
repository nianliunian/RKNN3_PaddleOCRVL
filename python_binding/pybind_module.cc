// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <cstring>
#include <mutex>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ── Include the demo source headers (from build tree copy) ─────
#include "paddleocr_vl.h"
#include "llm/rknn_paddleocr_vl_llm.h"
#include "vision/rknn_paddleocr_vl_vision.h"
#include "mlpar/rknn_paddleocr_vl_mlpar.h"
#include "common.h"
#include "image_utils.h"
#include "time_utils.h"

#include "Tokenizer.h"

namespace py = pybind11;

// ── Global LLM vars (defined in main.cc, re-defined here for binding) ──
const char* system_prompt  = "";
const char* prompt_prefix  = "<|begin_of_sentence|>User: ";
const char* prompt_postfix = "\nAssistant: ";

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 0.0f,
    .repeat_penalty = 1.1f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f
};

// Prompt templates
const char* DEFAULT_PROMPT  = "OCR:";
const char* TABLE_PROMPT    = "Table Recognition:";
const char* CHART_PROMPT    = "Chart Recognition:";
const char* FORMULA_PROMPT  = "Formula Recognition:";
const char* IMAGE_PROMPT    = "Image Recognition:";

#define MAX_NEW_TOKENS 1024
#define MAX_CONTEXT_LEN 1024

// ── Context that travels through callbacks ──────────────────────
struct PyBindingContext {
    Tokenizer* tokenizer;
    int embedding_fd;
    float16* embedding_data;
    int embedding_dim;
    int vocab_size;
    std::string collected_text;
    std::mutex text_mutex;
    int64_t first_token;
    bool first_decode;
    rknn_perf_metrics_t perf;
};

// ── Callbacks ───────────────────────────────────────────────────
static int py_result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    if (!userdata) return 0;
    PyBindingContext* ctx = (PyBindingContext*)userdata;
    Tokenizer* tokenizer = ctx->tokenizer;

    if (state == RKLLM_RUN_ERROR) {
        printf("Error occurred during inference\n");
        return 0;
    }
    else if (state == RKLLM_RUN_FINISH) {
        return 0;
    }
    else if (state == RKLLM_RUN_WAITING) {
        return 0;
    }
    else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED) {
        return 0;
    }
    else if (state == RKLLM_RUN_STOP) {
        return 0;
    }
    else if (state == RKLLM_RUN_NORMAL) {
        std::string piece;
        if (result->num_tokens > 1) {
            piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        } else {
            piece = tokenizer->TokenToPiece(result->token_ids[0]);
        }
        {
            std::lock_guard<std::mutex> lock(ctx->text_mutex);
            ctx->collected_text += piece;
        }
        if (ctx->first_decode) {
            ctx->first_token = getCurrentTimeUs();
            ctx->first_decode = false;
        }
    }
    return 0;
}

static int py_tokenizer_callback(void *userdata, const char *text, int32_t text_len, int32_t *tokens, int32_t n_tokens_max)
{
    if (!userdata) return 0;
    PyBindingContext* ctx = (PyBindingContext*)userdata;
    Tokenizer* tokenizer = ctx->tokenizer;
    int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);
    if (n_tokens <= 0) {
        printf("tokenizer failed for %s\n", text);
    }
    return n_tokens;
}

static int py_embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
    if (!userdata) return -1;
    PyBindingContext* ctx = (PyBindingContext*)userdata;
    if (len != num_tokens * ctx->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer\n");
        return -1;
    }
    for (int n = 0; n < (int)num_tokens; n++) {
        memcpy((unsigned char*)embed + n * ctx->embedding_dim * sizeof(float16),
               ctx->embedding_data + tokens[n] * ctx->embedding_dim,
               ctx->embedding_dim * sizeof(float16));
    }
    return 0;
}

// ── Python binding class ────────────────────────────────────────
class PaddleOCRVL {
public:
    PaddleOCRVL() : app_ctx_(nullptr), tokenizer_(nullptr), ctx_(nullptr),
                    vocab_size_(0), embedding_dim_(0), embedding_fd_(-1),
                    embedding_data_(nullptr), vision_embeds_(nullptr), img_embeds_(nullptr),
                    vision_embed_elems_(0), img_embed_elems_(0) {}

    ~PaddleOCRVL() {
        release();
    }

    void init(const std::string& model_dir,
              uint32_t vision_core_mask = 0x1,
              uint32_t mlpar_core_mask = 0x2,
              uint32_t llm_core_mask = 0x4,
              uint32_t model_width = 504,
              uint32_t model_height = 504)
    {
        // Build absolute paths
        std::string vision_model_path      = model_dir + "/PaddleOCR-vision.rknn";
        std::string vision_weight_path     = model_dir + "/PaddleOCR-vision.weight";
        std::string position_embedding_path = model_dir + "/vision/position_embedding_model.bin";
        std::string llm_model_path         = model_dir + "/PaddleOCR-llm.rknn";
        std::string llm_weight_path        = model_dir + "/PaddleOCR-llm.weight";
        std::string tokenizer_path         = model_dir + "/llm/PaddleOCR-llm.tokenizer.gguf";
        std::string embedding_path         = model_dir + "/llm/PaddleOCR-llm.embed.bin";
        std::string mlpar_model_path       = model_dir + "/PaddleOCR-vision-mlp_AR.rknn";
        std::string mlpar_weight_path      = model_dir + "/PaddleOCR-vision-mlp_AR.weight";

        // Load tokenizer
        tokenizer_ = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path.c_str());
        if (!tokenizer_) {
            throw std::runtime_error("Failed to load tokenizer: " + tokenizer_path);
        }

        VocabInfo vocab_info;
        tokenizer_->GetVocabInfo(&vocab_info);
        vocab_size_ = vocab_info.vocab_size;

        // Load embedding via mmap
        embedding_fd_ = open(embedding_path.c_str(), O_RDONLY);
        if (embedding_fd_ == -1) {
            throw std::runtime_error("Failed to open embedding file: " + embedding_path);
        }
        struct stat emb_st;
        fstat(embedding_fd_, &emb_st);
        embedding_data_ = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_fd_, 0);
        if (embedding_data_ == MAP_FAILED || embedding_data_ == nullptr) {
            close(embedding_fd_);
            embedding_fd_ = -1;
            throw std::runtime_error("Failed to mmap embedding file: " + embedding_path);
        }
        embedding_dim_ = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

        // Create PyBindingContext for callbacks
        ctx_ = new PyBindingContext();
        ctx_->tokenizer = tokenizer_;
        ctx_->embedding_fd = embedding_fd_;
        ctx_->embedding_data = embedding_data_;
        ctx_->embedding_dim = embedding_dim_;
        ctx_->vocab_size = vocab_size_;
        ctx_->collected_text.clear();
        ctx_->first_token = 0;
        ctx_->first_decode = true;

        // Setup LLM params
        rknn3_llm_param params;
        memset(&params, 0, sizeof(rknn3_llm_param));
        params.logits_name              = (char*)"logits";
        params.max_context_len          = MAX_CONTEXT_LEN;
        params.sampling_param           = SAMPLE_PARAMS;
        params.vocab_info.vocab_size    = vocab_info.vocab_size;
        params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
        params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
        memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
        memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
        params.vocab_info.linefeed_id   = vocab_info.linefeed_id;

        // Setup callback
        RKLLMCallback callback;
        memset(&callback, 0, sizeof(RKLLMCallback));
        callback.result_callback    = py_result_callback;
        callback.result_userdata    = ctx_;
        callback.tokenizer_callback = py_tokenizer_callback;
        callback.tokenizer_userdata = ctx_;
        callback.embed_callback     = py_embed_callback;
        callback.embed_userdata     = ctx_;

        // Create app context
        app_ctx_ = new rknn_app_context_t();
        memset(app_ctx_, 0, sizeof(rknn_app_context_t));
        app_ctx_->model_width = model_width;
        app_ctx_->model_height = model_height;

        // Init all sub-models
        int ret = init_paddleocr_vl_model(
            app_ctx_,
            llm_model_path.c_str(), llm_weight_path.c_str(),
            vision_model_path.c_str(), vision_weight_path.c_str(),
            position_embedding_path.c_str(),
            mlpar_model_path.c_str(), mlpar_weight_path.c_str(),
            &params, 1, callback,
            vision_core_mask, mlpar_core_mask, llm_core_mask
        );
        if (ret != 0) {
            throw std::runtime_error("Failed to init paddleocr_vl model, ret=" + std::to_string(ret));
        }

        // Allocate vision embeds
        vision_embed_elems_ = 1;
        for (size_t i = 0; i < app_ctx_->vision.embeds_ndims; i++) {
            vision_embed_elems_ *= app_ctx_->vision.embeds_shape[i];
        }
        vision_embeds_ = (float16*)malloc(vision_embed_elems_ * sizeof(float16));

        // Allocate image embeds
        img_embed_elems_ = 1;
        for (size_t i = 0; i < app_ctx_->mlpar.embeds_ndims; i++) {
            img_embed_elems_ *= app_ctx_->mlpar.embeds_shape[i];
        }
        img_embeds_ = (float16*)malloc(img_embed_elems_ * sizeof(float16));
    }

    py::dict run(const std::string& image_path,
                 const std::string& prompt = "OCR:",
                 const std::string& prompt_type = "ocr")
    {
        if (!app_ctx_) {
            throw std::runtime_error("Model not initialized. Call init() first.");
        }

        // Reset collection state
        ctx_->collected_text.clear();
        ctx_->first_token = 0;
        ctx_->first_decode = true;

        // Select prompt template
        const char* selected_prompt = DEFAULT_PROMPT;
        if (prompt_type == "table") selected_prompt = TABLE_PROMPT;
        else if (prompt_type == "chart") selected_prompt = CHART_PROMPT;
        else if (prompt_type == "formula") selected_prompt = FORMULA_PROMPT;
        else if (prompt_type == "image") selected_prompt = IMAGE_PROMPT;
        else if (prompt != DEFAULT_PROMPT && prompt != TABLE_PROMPT && prompt != CHART_PROMPT &&
                 prompt != FORMULA_PROMPT && prompt != IMAGE_PROMPT) {
            // user provided custom prompt, use as-is
            selected_prompt = prompt.c_str();
        }

        // Read image
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        int ret = read_image(image_path.c_str(), &src_image);
        if (ret != 0) {
            throw std::runtime_error("Failed to read image: " + image_path);
        }

        // Build multimodal tensor
        rknn3_llm_multimodal_tensor tensor;
        memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));
        std::string prompt_with_image = "<image> " + std::string(selected_prompt);
        tensor.name = "input_embeds";
        tensor.prompt = prompt_with_image.c_str();
        tensor.image.image_embed = img_embeds_;
        if (app_ctx_->mlpar.embeds_ndims == 2) {
            tensor.image.n_image_tokens = app_ctx_->mlpar.embeds_shape[0];
            tensor.image.n_image = 1;
        } else {
            tensor.image.n_image_tokens = app_ctx_->mlpar.embeds_shape[1];
            tensor.image.n_image = app_ctx_->mlpar.embeds_shape[0];
        }
        tensor.image.image_width  = app_ctx_->vision.model_width;
        tensor.image.image_height = app_ctx_->vision.model_height;
        tensor.image.image_start  = "<|IMAGE_START|>";
        tensor.image.image_end    = "<|IMAGE_END|>";
        tensor.image.image_content = "<|IMAGE_PLACEHOLDER|>";
        tensor.enable_thinking = false;

        // Run inference
        rknn_perf_metrics_t perf;
        int n_inputs = 1;
        ret = inference_paddleocr_vl_model(
            app_ctx_, &src_image, vision_embeds_, img_embeds_,
            tensor, n_inputs, &perf
        );
        if (ret != 0) {
            throw std::runtime_error("Inference failed, ret=" + std::to_string(ret));
        }

        // Free image buffer
        if (src_image.virt_addr != nullptr) {
            free(src_image.virt_addr);
        }

        // Collect result
        std::string text;
        {
            std::lock_guard<std::mutex> lock(ctx_->text_mutex);
            text = ctx_->collected_text;
        }

        // Compute performance data
        float ttft_us = (float)(ctx_->first_token - perf.llm_start_time);
        float prefill_ms = ttft_us / 1000.0f;
        float prefill_tpt = perf.n_prefill_tokens == 0 ? 0.0f : prefill_ms / perf.n_prefill_tokens;
        float prefill_tps = perf.n_prefill_tokens == 0 ? 0.0f : 1000.0f / prefill_ms * perf.n_prefill_tokens;

        float decode_time_us = (float)(perf.llm_end_time - ctx_->first_token);
        float decode_ms = decode_time_us / 1000.0f;
        float decode_tpt = perf.n_decode_tokens == 0 ? 0.0f : decode_ms / perf.n_decode_tokens;
        float decode_tps = perf.n_decode_tokens == 0 ? 0.0f : 1000.0f / decode_ms * perf.n_decode_tokens;

        // Build return dict
        py::dict result;
        result["text"] = text;
        py::dict metrics;
        metrics["vision_latency"] = (double)perf.vision_latency / 1000.0;
        metrics["prefill_latency"] = (double)prefill_ms;
        metrics["decode_latency"] = (double)decode_ms;
        metrics["ttft"] = (double)ttft_us / 1000.0;
        metrics["prefill_tokens"] = perf.n_prefill_tokens;
        metrics["decode_tokens"] = perf.n_decode_tokens;
        metrics["tpt"] = (double)((perf.n_decode_tokens + perf.n_prefill_tokens) > 0
                               ? (ttft_us + decode_time_us) / 1000.0 / (perf.n_decode_tokens + perf.n_prefill_tokens)
                               : 0.0);
        metrics["tps"] = (double)((perf.n_decode_tokens + perf.n_prefill_tokens) > 0
                               ? 1000.0 / (ttft_us + decode_time_us) * (perf.n_decode_tokens + perf.n_prefill_tokens)
                               : 0.0);
        metrics["prefill_tpt"] = (double)prefill_tpt;
        metrics["prefill_tps"] = (double)prefill_tps;
        metrics["decode_tpt"] = (double)decode_tpt;
        metrics["decode_tps"] = (double)decode_tps;
        result["metrics"] = metrics;

        return result;
    }

    void release() {
        if (app_ctx_) {
            release_paddleocr_vl_model(app_ctx_);
            delete app_ctx_;
            app_ctx_ = nullptr;
        }
        if (tokenizer_) {
            delete tokenizer_;
            tokenizer_ = nullptr;
        }
        if (ctx_) {
            delete ctx_;
            ctx_ = nullptr;
        }
        if (vision_embeds_) {
            free(vision_embeds_);
            vision_embeds_ = nullptr;
            vision_embed_elems_ = 0;
        }
        if (img_embeds_) {
            free(img_embeds_);
            img_embeds_ = nullptr;
            img_embed_elems_ = 0;
        }
        if (embedding_data_ != nullptr && embedding_data_ != MAP_FAILED) {
            if (embedding_fd_ != -1) {
                struct stat emb_st;
                fstat(embedding_fd_, &emb_st);
                munmap((void*)embedding_data_, emb_st.st_size);
            }
            embedding_data_ = nullptr;
        }
        if (embedding_fd_ != -1) {
            close(embedding_fd_);
            embedding_fd_ = -1;
        }
    }

private:
    rknn_app_context_t* app_ctx_;
    Tokenizer* tokenizer_;
    PyBindingContext* ctx_;
    int vocab_size_;
    int embedding_dim_;
    int embedding_fd_;
    float16* embedding_data_;
    float16* vision_embeds_;
    float16* img_embeds_;
    size_t vision_embed_elems_;
    size_t img_embed_elems_;
};

// ── Module definition ───────────────────────────────────────────
PYBIND11_MODULE(paddleocr_vl, m) {
    m.doc() = "PaddleOCR-VL Python binding for RKNN3 (pybind11)";

    py::class_<PaddleOCRVL>(m, "PaddleOCRVL")
        .def(py::init<>())
        .def("init", &PaddleOCRVL::init,
             py::arg("model_dir") = "model",
             py::arg("vision_core_mask") = 0x1,
             py::arg("mlpar_core_mask") = 0x2,
             py::arg("llm_core_mask") = 0x4,
             py::arg("model_width") = 504,
             py::arg("model_height") = 504,
             "Initialize PaddleOCR-VL model from model_dir.")
        .def("run", &PaddleOCRVL::run,
             py::arg("image_path"),
             py::arg("prompt") = "OCR:",
             py::arg("prompt_type") = "ocr",
             "Run inference on an image. Returns dict with text and metrics.")
        .def("release", &PaddleOCRVL::release,
             "Release model resources and free memory.");
}
