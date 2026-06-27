"""VLMRunner: refactored from infer_simular.py for repeatable invocation.

Key changes from infer_simular.py:
- Resources (RKNN models, embeds, tokenizer, rope_cache) loaded once in __init__
- run() accepts PIL.Image / ndarray / path (no temp file I/O)
- prompt_label validated before inference
- prepare_vlm_input refactored to accept PIL.Image directly
- KV cache reset between calls (via init_runtime re-call if API lacks reset)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Set HF mirror before importing transformers
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

logger = logging.getLogger(__name__)

SUPPORTED_PROMPTS = {"ocr", "table", "formula", "chart"}
DEFAULT_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

# Model constants (from infer_simular.py)
_VOCAB_SIZE = 103424
_SEQ_LEN = 128
_PATCH_SIZE = 14
_IMAGE_TOKEN_ID = 100295
_MAX_NEW_TOKENS = 1024


def _llm_logitsprocessor(input_ids, logits, args=None):
    args = args or {}
    temperature = args.get("temperature", 1.0)
    top_k = args.get("top_k", 1)
    top_p = args.get("top_p", 0.9)
    repetition_penalty = args.get("repeat_penalty", 1.0)
    do_sample = args.get("do_sample", False)

    warpers = [
        TemperatureLogitsWarper(temperature),
        RepetitionPenaltyLogitsProcessor(repetition_penalty)
        if input_ids is not None
        else None,
        TopKLogitsWarper(top_k=top_k),
        TopPLogitsWarper(top_p=top_p),
    ]
    for warper in warpers:
        if warper is not None:
            logits = warper(input_ids=input_ids, scores=logits)
    probs = torch.softmax(logits, dim=-1)
    if do_sample:
        next_token = torch.multinomial(probs, num_samples=1)[0]
    else:
        next_token = torch.argmax(probs, dim=-1)
    return next_token.numpy()


def _find_best_size(
    original_width: int,
    original_height: int,
    patch_size: int = 28,
    target_size: int = 196,
) -> tuple[int, int]:
    import math

    original_ratio = original_width / original_height
    factors = []
    for a in range(1, int(math.sqrt(target_size)) + 1):
        if target_size % a == 0:
            b = target_size // a
            factors.append((a, b))
            if a != b:
                factors.append((b, a))
    best_ratio_diff = float("inf")
    best_pair = (1, target_size)
    for a, b in factors:
        diff = abs((a / b) - original_ratio)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_pair = (a, b)
    return best_pair[0] * patch_size, best_pair[1] * patch_size


def _interpolate_pos_encoding(
    height: int,
    width: int,
    position_embedding_path: str,
    is_after_patchify: bool = False,
) -> torch.Tensor:
    dim = 1152
    new_height = height if is_after_patchify else height // 14
    new_width = width if is_after_patchify else width // 14
    patch_pos_embed = np.fromfile(
        position_embedding_path, dtype=np.float32
    )
    patch_pos_embed = torch.from_numpy(patch_pos_embed)
    patch_pos_embed = patch_pos_embed.reshape(1, 1152, 27, 27)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed


def _get_rope_emb(image_grid_thw):
    split_hids, split_wids = [], []
    for t, h, w in image_grid_thw:
        image_pids = torch.arange(t * h * w) % (h * w)
        sample_hids = image_pids // w
        sample_wids = image_pids % w
        split_hids.append(sample_hids)
        split_wids.append(sample_wids)
    width_position_ids = torch.concat(split_wids, dim=0)
    height_position_ids = torch.concat(split_hids, dim=0)
    pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
    max_grid_size = pids.max() + 1

    dim = 36
    theta = 10000.0
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    )
    seq = torch.arange(
        max_grid_size, device=inv_freq.device, dtype=inv_freq.dtype
    )
    rope_emb_max_grid = torch.outer(seq, inv_freq)
    rope_emb = rope_emb_max_grid[pids].flatten(1)
    rope_emb = rope_emb.repeat(1, 2)
    return rope_emb


class VLMRunner:
    """Wraps PaddleOCR-VL RKNN models for repeatable VLM inference."""

    def __init__(
        self,
        vision_rknn_path: str,
        mlpar_rknn_path: str,
        llm_rknn_path: str,
        embed_path: str,
        llm_config_path: str,
        tokenizer_path: str,
        position_embedding_path: str,
        vision_module_path: str,
        target: bool = False,
        device_id: str | None = None,
        max_new_tokens: int = _MAX_NEW_TOKENS,
    ):
        # Import locally so module import doesn't fail without RKNN
        from rknn.api import RKNN

        self._RKNN = RKNN
        self.target = target
        self.device_id = device_id
        self.max_new_tokens = max_new_tokens
        self.position_embedding_path = position_embedding_path
        self.llm_config_path = llm_config_path
        self.vision_module_path = vision_module_path

        # Load img_h, img_w from config
        self.img_h, self.img_w = self._load_config(llm_config_path)

        # Load three RKNN models
        self.vision_rknn = self._load_rknn(
            vision_rknn_path, "vision"
        )
        self.mlpar_rknn = self._load_rknn(
            mlpar_rknn_path, "mlpar"
        )
        self.llm_rknn = self._load_rknn(
            llm_rknn_path, "llm"
        )
        # Remember paths for KV cache reset
        self._llm_rknn_path = llm_rknn_path

        # Load embeds
        self.embeds_data = np.fromfile(
            embed_path, dtype=np.float16
        ).reshape(_VOCAB_SIZE, -1)

        # Load tokenizer + processor (cached)
        logger.info("Loading tokenizer/processor from %s", tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        # Cache rope_cache
        self.rope_cache = self.llm_rknn.query("QUERY_ROPE_CACHE")
        logger.info("VLMRunner initialized")

    def _load_config(self, config_path: str) -> tuple[int, int]:
        if not os.path.exists(config_path):
            logger.warning(
                "Config not found %s, using default 504x504", config_path
            )
            return 504, 504
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ("img_h", "img_w"):
            if key not in cfg:
                raise KeyError(f"Missing key {key!r} in config {config_path}")
        return cfg["img_h"], cfg["img_w"]

    def _load_rknn(self, model_path: str, name: str):
        rknn = self._RKNN(verbose=False)
        rknn.config(
            target_platform="rk1820",
            quantized_dtype="w4a16",
            quantized_algorithm="grq",
            quantized_method="group32",
        )
        weight_path = model_path.replace(".rknn", ".weight")
        ret = rknn.load_rknn(model_path, weight_path, load_ctx=True)
        if ret != 0:
            raise RuntimeError(
                f"Load {name} RKNN failed (ret={ret}): {model_path}"
            )
        if self.target:
            ret = rknn.init_runtime(
                target="rk1820", core_mask=0xff, device_id=self.device_id
            )
        else:
            ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(
                f"Init {name} runtime failed (ret={ret})"
            )
        logger.info("Loaded %s RKNN: %s", name, model_path)
        return rknn

    @staticmethod
    def _validate_prompt_label(prompt_label: str) -> None:
        if prompt_label not in SUPPORTED_PROMPTS:
            raise ValueError(
                f"Unsupported prompt_label: {prompt_label!r}. "
                f"Must be one of {sorted(SUPPORTED_PROMPTS)}"
            )

    def _to_pil_image(self, image: str | np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            # ndarray is likely BGR (cv2) or RGB; assume RGB if 3 channels
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 ndarray, got shape {image.shape}")
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            "Expected str (path), np.ndarray, or PIL.Image.Image"
        )

    def _prepare_vlm_input(
        self, image: Image.Image, prompt_label: str
    ) -> tuple[dict, Any]:
        """Refactored from infer_simular.py prepare_vlm_input.

        Accepts PIL.Image directly instead of image_path.
        """
        old_w, old_h = image.size
        new_w, new_h = _find_best_size(
            old_w, old_h,
            target_size=self.img_h // _PATCH_SIZE // 2
            * self.img_w // _PATCH_SIZE // 2,
        )
        logger.debug(
            "image size: src(%d,%d) -> new(%d,%d)",
            old_w, old_h, new_w, new_h,
        )
        image = image.resize(
            (new_w, new_h), resample=Image.Resampling.BICUBIC
        )

        # Import BasicImageTransform locally to avoid hard dep at module import
        import sys
        sys.path.insert(0, self.vision_module_path)
        from export_vision import BasicImageTransform

        image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
        )
        image_arr = np.array(image)
        pixel_values = image_transform(image_arr).to(torch.float32)
        grid_t, grid_h, grid_w = 1, new_h // _PATCH_SIZE, new_w // _PATCH_SIZE
        patches = pixel_values.reshape(
            grid_t, 1, 3, grid_h, _PATCH_SIZE, grid_w, _PATCH_SIZE
        ).numpy()
        patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, 3, _PATCH_SIZE, _PATCH_SIZE
        )
        pixel_values = torch.from_numpy(flatten_patches).type(torch.float32).unsqueeze(0)

        grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64)
        position_embedding = _interpolate_pos_encoding(
            grid_h, grid_w, self.position_embedding_path, True
        )
        rope_emb = _get_rope_emb(grid_thw)

        inputs = [
            pixel_values.numpy(),
            position_embedding.numpy(),
            rope_emb.numpy(),
        ]
        data_format = ["nchw"] * len(inputs)
        vision_embed = self.vision_rknn.inference(
            inputs, data_format, accuracy_analysis=False
        )[0]

        t, h, w = grid_thw[0]
        from einops import rearrange

        vision_embed = rearrange(
            vision_embed,
            "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
            t=t, h=h // 2, p1=2, w=w // 2, p2=2,
        )
        inputs = [vision_embed]
        data_format = ["nchw"] * len(inputs)
        image_embeds = self.mlpar_rknn.inference(
            inputs, data_format, accuracy_analysis=False
        )[0]

        query = DEFAULT_PROMPTS[prompt_label]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        proc_inputs = self.processor(
            image_arr, text=text, return_tensors="pt", format=True
        )
        input_ids = proc_inputs["input_ids"].cpu().numpy().astype(np.int64)
        inputs_embeds = self.embeds_data[input_ids].astype(np.float32)

        n_image_tokens = (input_ids == _IMAGE_TOKEN_ID).sum()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={n_image_features}"
            )

        mask = input_ids == _IMAGE_TOKEN_ID
        batch_idx, seq_idx = np.where(mask)
        inputs_embeds[batch_idx, seq_idx] = image_embeds.astype(
            inputs_embeds.dtype
        )
        proc_inputs["inputs_embeds"] = inputs_embeds
        proc_inputs["attention_mask"] = (
            proc_inputs["attention_mask"].cpu().numpy().astype(np.float32)
        )
        return proc_inputs, input_ids

    def _reset_kv_cache(self) -> None:
        """Reset KV cache before each run.

        Uses rknn.kvcache_controller.clear_kvcache_status() — cheap O(1)
        operation that wipes the KV state without reloading the model.

        Previously this released + reloaded the 243MB LLM RKNN model per
        block, causing ~10s delay per VLM call and excessive log spam.
        """
        try:
            self.llm_rknn.kvcache_controller.clear_kvcache_status()
        except Exception as e:
            logger.warning("KV cache reset failed: %s", e)

    def run(
        self,
        image: str | np.ndarray | Image.Image,
        prompt_label: str,
    ) -> str:
        """Run VLM inference on an image with a given prompt label.

        Args:
            image: path / ndarray / PIL.Image
            prompt_label: one of {ocr, table, formula, chart}

        Returns:
            VLM-generated text string.
        """
        self._reset_kv_cache()
        self._validate_prompt_label(prompt_label)
        pil_image = self._to_pil_image(image)

        inputs, input_ids_arr = self._prepare_vlm_input(
            pil_image, prompt_label
        )

        embeds = inputs["inputs_embeds"]
        token_ids = input_ids_arr
        input_seq_len = token_ids.shape[1]
        logger.info("VLM input seq len: %d", input_seq_len)

        eos_token_ids = [self.tokenizer.eos_token_id]
        generate_ids: list = []

        # Prefill
        if _SEQ_LEN >= input_seq_len:
            inputs_embeds = np.zeros(
                (1, _SEQ_LEN, embeds.shape[-1]), dtype=np.float32
            )
            attention_mask = np.zeros((1, _SEQ_LEN), dtype=np.float32)
            inputs_embeds[:, :input_seq_len, :] = embeds
            attention_mask[:, :input_seq_len] = 1
            num_logits_to_keep = np.array(
                [input_seq_len - 1], dtype=np.int32
            )
            attention_inputs, _ = (
                self.llm_rknn.kvcache_controller.generate_kvcache_control_tensors(
                    input_seq_len
                )
            )
            prefill_inputs = (
                [inputs_embeds, attention_mask, num_logits_to_keep]
                + self.rope_cache
                + attention_inputs[0]
            )
            data_format = ["nchw"] * len(prefill_inputs)
            prefill_logits = self.llm_rknn.inference(
                prefill_inputs, data_format, accuracy_analysis=False
            )[0]
        else:
            attention_inputs, _ = (
                self.llm_rknn.kvcache_controller.generate_kvcache_control_tensors(
                    input_seq_len
                )
            )
            prefill_logits = None
            for i, seq_len in enumerate(range(0, input_seq_len, _SEQ_LEN)):
                inputs_embeds = np.zeros(
                    (1, _SEQ_LEN, embeds.shape[-1]), dtype=np.float32
                )
                attention_mask = np.zeros(
                    (1, _SEQ_LEN), dtype=np.float32
                )
                curr_len = min(_SEQ_LEN, input_seq_len - seq_len)
                inputs_embeds[:, :curr_len, :] = embeds[
                    :, seq_len : seq_len + curr_len, :
                ]
                attention_mask[:, :curr_len] = 1
                num_logits_to_keep = np.array(
                    [curr_len - 1], dtype=np.int32
                )
                prefill_inputs = (
                    [inputs_embeds, attention_mask, num_logits_to_keep]
                    + self.rope_cache
                    + attention_inputs[i]
                )
                data_format = ["nchw"] * len(prefill_inputs)
                prefill_logits = self.llm_rknn.inference(
                    prefill_inputs, data_format, accuracy_analysis=False
                )[0]

        next_token = _llm_logitsprocessor(
            torch.from_numpy(token_ids),
            torch.from_numpy(prefill_logits).reshape(1, -1),
        )
        generate_ids.append(next_token[0])

        # Decoder loop
        for i in range(self.max_new_tokens - 1):
            input_ids_step = np.expand_dims(next_token, axis=0).astype(np.int64)
            inputs_embeds = self.embeds_data[input_ids_step].astype(np.float32)
            attention_mask = np.expand_dims(
                np.array([1]), axis=0
            ).astype(np.float32)
            num_logits_to_keep = np.array([0], dtype=np.int32)
            token_ids = np.concatenate(
                (token_ids, input_ids_step), axis=1
            )
            attention_inputs, _ = (
                self.llm_rknn.kvcache_controller.generate_kvcache_control_tensors(
                    1
                )
            )
            decoder_inputs = (
                [inputs_embeds, attention_mask, num_logits_to_keep]
                + self.rope_cache
                + attention_inputs[0]
            )
            data_format = ["nchw"] * len(decoder_inputs)
            decoder_logits = self.llm_rknn.inference(
                decoder_inputs, data_format
            )[0]
            next_token = _llm_logitsprocessor(
                torch.from_numpy(token_ids),
                torch.from_numpy(decoder_logits).reshape(1, -1),
            )
            generate_ids.append(next_token[0])
            if next_token[-1] in eos_token_ids:
                logger.info("VLM inference completed (EOS at step %d)", i + 1)
                break
            if (i + 1) % 20 == 0:
                logger.debug(
                    "VLM step %d: %s",
                    i + 1,
                    self.tokenizer.decode(
                        generate_ids, skip_special_tokens=True
                    )[:100],
                )

        response = self.tokenizer.decode(
            generate_ids, skip_special_tokens=True
        )
        return response

    def close(self) -> None:
        """Release RKNN resources."""
        for name, rknn in [
            ("vision", getattr(self, "vision_rknn", None)),
            ("mlpar", getattr(self, "mlpar_rknn", None)),
            ("llm", getattr(self, "llm_rknn", None)),
        ]:
            if rknn is not None:
                try:
                    rknn.release()
                except Exception as e:
                    logger.warning("Failed to release %s RKNN: %s", name, e)
