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
hdr = "  {:<12} {:>15} {:>8} {:>18} {:>18}".format("Stage", "Total Time (ms)", "Tokens", "Time/Token (ms)", "Tokens/sec")
print(hdr)
print("-" * 90)

prefill_ms = m["prefill_latency"]
prefill_tokens = int(m["prefill_tokens"])
prefill_tpt = m["prefill_tpt"]
prefill_tps = m["prefill_tps"]
print("  {:<12} {:>15.2f} {:>8} {:>18.2f} {:>18.2f}".format("Prefill", prefill_ms, prefill_tokens, prefill_tpt, prefill_tps))

decode_ms = m["decode_latency"]
decode_tokens = int(m["decode_tokens"])
decode_tpt = m["decode_tpt"]
decode_tps = m["decode_tps"]
print("  {:<12} {:>15.2f} {:>8} {:>18.2f} {:>18.2f}".format("Generate", decode_ms, decode_tokens, decode_tpt, decode_tps))

print("-" * 90)
print("  TTFT: {:.2f} ms".format(m["ttft"]))
print("  Vision latency: {:.2f} ms".format(m["vision_latency"]))
print("  Total tokens: {}".format(prefill_tokens + decode_tokens))
print("  Overall TPS: {:.2f} tokens/s".format(m["tps"]))
print("=" * 90)

model.release()
