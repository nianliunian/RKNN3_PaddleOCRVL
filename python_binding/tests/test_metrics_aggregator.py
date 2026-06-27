"""Tests for MetricsAggregator."""
from pipeline.metrics_aggregator import MetricsAggregator


def _block_metrics(vision=10.0, prefill=20.0, prefill_tokens=5,
                   decode=100.0, decode_tokens=10):
    return {
        "vision_latency": vision,
        "prefill_latency": prefill,
        "prefill_tokens": prefill_tokens,
        "decode_latency": decode,
        "decode_tokens": decode_tokens,
    }


class TestMetricsAggregator:
    def test_empty_build_has_zero_totals(self):
        agg = MetricsAggregator()
        m = agg.build(layout_ms=50.0, layout_boxes=0, blocks_skipped=0)
        assert m["layout_ms"] == 50.0
        assert m["layout_boxes"] == 0
        assert m["vlm_blocks_run"] == 0
        assert m["vlm_blocks_skipped"] == 0
        assert m["vision_latency_total_ms"] == 0.0
        assert m["prefill_latency_total_ms"] == 0.0
        assert m["prefill_tokens_total"] == 0
        assert m["decode_latency_total_ms"] == 0.0
        assert m["decode_tokens_total"] == 0
        assert m["vlm_total_ms"] == 0.0
        assert m["page_total_ms"] == 50.0
        assert m["vlm_tps_avg"] == 0.0

    def test_single_block_accumulated(self):
        agg = MetricsAggregator()
        agg.add_block(_block_metrics())
        m = agg.build(layout_ms=50.0, layout_boxes=1, blocks_skipped=0)
        assert m["vlm_blocks_run"] == 1
        assert m["vision_latency_total_ms"] == 10.0
        assert m["prefill_latency_total_ms"] == 20.0
        assert m["prefill_tokens_total"] == 5
        assert m["decode_latency_total_ms"] == 100.0
        assert m["decode_tokens_total"] == 10
        # vlm_total = vision + prefill + decode per block
        assert m["vlm_total_ms"] == 130.0
        assert m["page_total_ms"] == 180.0
        # tps = decode_tokens / (decode_ms / 1000)
        assert m["vlm_tps_avg"] == 10 / 0.1

    def test_multi_block_sum(self):
        agg = MetricsAggregator()
        agg.add_block(_block_metrics(vision=10, prefill=20, decode=100))
        agg.add_block(_block_metrics(vision=5, prefill=10, decode=50))
        m = agg.build(layout_ms=0.0, layout_boxes=2, blocks_skipped=1)
        assert m["vlm_blocks_run"] == 2
        assert m["vlm_blocks_skipped"] == 1
        assert m["vision_latency_total_ms"] == 15.0
        assert m["prefill_latency_total_ms"] == 30.0
        assert m["decode_latency_total_ms"] == 150.0
        assert m["vlm_total_ms"] == 195.0

    def test_tps_avg_handles_zero_decode_latency(self):
        agg = MetricsAggregator()
        agg.add_block(_block_metrics(decode=0.0, decode_tokens=5))
        m = agg.build(layout_ms=0.0, layout_boxes=1, blocks_skipped=0)
        assert m["vlm_tps_avg"] == 0.0
