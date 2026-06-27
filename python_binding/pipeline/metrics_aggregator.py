"""Accumulates per-block VLM metrics into a page-level dict."""
from __future__ import annotations


class MetricsAggregator:
    """Sums per-block VLM metrics; final build adds layout timing + derived fields."""

    def __init__(self):
        self._block_count = 0
        self._vision_total = 0.0
        self._prefill_total = 0.0
        self._prefill_tokens_total = 0
        self._decode_total = 0.0
        self._decode_tokens_total = 0

    def add_block(self, block_metrics: dict) -> None:
        self._block_count += 1
        self._vision_total += float(block_metrics.get("vision_latency", 0.0))
        self._prefill_total += float(block_metrics.get("prefill_latency", 0.0))
        self._prefill_tokens_total += int(block_metrics.get("prefill_tokens", 0))
        self._decode_total += float(block_metrics.get("decode_latency", 0.0))
        self._decode_tokens_total += int(block_metrics.get("decode_tokens", 0))

    def build(
        self,
        layout_ms: float,
        layout_boxes: int,
        blocks_skipped: int,
    ) -> dict:
        vlm_total_ms = self._vision_total + self._prefill_total + self._decode_total
        if self._decode_total > 0:
            tps_avg = self._decode_tokens_total / (self._decode_total / 1000.0)
        else:
            tps_avg = 0.0
        # ttft per block equals prefill_latency (see pybind_module.cc:350-351);
        # report the per-block average so it matches the .so single-inference semantics.
        ttft_avg = self._prefill_total / self._block_count if self._block_count else 0.0
        return {
            "layout_ms": layout_ms,
            "layout_boxes": layout_boxes,
            "vlm_blocks_run": self._block_count,
            "vlm_blocks_skipped": blocks_skipped,
            "vision_latency_total_ms": self._vision_total,
            "prefill_latency_total_ms": self._prefill_total,
            "prefill_tokens_total": self._prefill_tokens_total,
            "decode_latency_total_ms": self._decode_total,
            "decode_tokens_total": self._decode_tokens_total,
            "ttft_avg_ms": ttft_avg,
            "vlm_total_ms": vlm_total_ms,
            "page_total_ms": layout_ms + vlm_total_ms,
            "vlm_tps_avg": tps_avg,
        }
