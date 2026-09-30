# Benchmark results

CPU rows, Intel Core Ultra 7 256V on AC power, each runtime in its own process
with a settle gap between them, reporting the fastest observed time. Library
rows are 10 rounds x 12 images. Engine rows are 5 rounds x 4 passes, which is
the same protocol scaled to a runtime that takes seconds rather than
milliseconds per image. GPU rows, Colab Tesla T4. mAP is scored over the same
49 test images with one shared implementation.

| Runtime | Precision | Best latency | Median | mAP@0.5 |
|---------|-----------|--------------|--------|---------|
| PyTorch (Ultralytics) | FP32 | 42.4 ms | 58.5 ms | 0.8859 |
| ONNX Runtime | FP32 | 24.1 ms | 28.4 ms | 0.8859 |
| ONNX Runtime | INT8 | 30.8 ms | 35.8 ms | 0.8556 |
| OpenVINO | FP32 | 28.5 ms | 33.3 ms | 0.8859 |
| OpenVINO | INT8 | 13.9 ms | 16.4 ms | 0.8089 |
| Custom C++ engine, scalar | FP32 arithmetic | 3105.9 ms | 3127.6 ms | 0.8836 |
| Custom C++ engine, scalar | INT8 | 3578.7 ms | 3776.1 ms | 0.8826 |
| Custom C++ engine, AVX-VNNI | INT8 | 233.5 ms | 238.7 ms | 0.8826 |
| Custom C++ engine (per-tensor), scalar | INT8 | 3418.1 ms | 3461.4 ms | 0.7680 |
| PyTorch (Ultralytics), T4 | FP32 | 8.0 ms | 8.1 ms | 0.8859 |
| TensorRT, T4, default export | INT8 | 4.4 ms | 5.1 ms | 0.0833 |
| TensorRT, T4, convolutions only | INT8 | 4.9 ms | 6.4 ms | 0.8656 |
| Custom C++ engine, CUDA GEMM, T4 | FP32 arithmetic | 295.2 ms | 365.1 ms | 0.8826 |
| Custom C++ engine, CUDA dp4a GEMM, T4 | INT8 | 290.4 ms | 290.4 ms | 0.8826 |
| Custom C++ engine, CUDA dp4a GEMM fused, T4 | INT8 | 62.1 ms | 63.3 ms | 0.8826 |

## What the GPU row does and does not measure

The three CUDA rows share one kernel: a 2D register-tiled GEMM from
[cuda-gemm-from-scratch](https://github.com/rachitj27/cuda-gemm-from-scratch),
generalized to the shapes this network produces and then ported to INT8 with
`__dp4a`. The convolution is genuinely fast, and for two of the three rows that
barely shows up in the latency.

| | GEMM per image | Rate | End to end |
|---|---|---|---|
| AVX-VNNI, one laptop core | 233.5 ms | 17.3 GMAC/s | 233.5 ms |
| CUDA FP32 GEMM | 10.63 ms | 380 GMAC/s | 295.2 ms |
| CUDA INT8 dp4a GEMM | 2.55 ms | 1588 GMAC/s | 290.4 ms |
| CUDA INT8, epilogue fused | 2.55 ms | 1588 GMAC/s | 62.1 ms |

Convolution was effectively the entire runtime before any of this. Making it 92
times faster moved the end-to-end number by nothing at all, because the work
around it -- the FP32 epilogue, requantization, and the concat, pooling,
upsample, DFL decode and NMS -- was all still on the CPU. Measured on the
unfused INT8 row, the GEMM was 0.9% of the wall clock.

Fusing the epilogue and the requantization onto the device is what produced the
speedup. It also shrinks the transfer back to the host fourfold, since the copy
carries INT8 rather than int32 accumulators. Of the 69 ms in a profiled run,
22 ms is on the GPU and 47 ms is what remains on the CPU.

A note on the comparison: the 62.1 ms and the 233.5 ms come from different
machines. The GPU rows run their host work on a Colab Xeon, which the same
scalar kernel shows is about 2.85 times slower than the laptop the CPU rows
were measured on, so the 3.8x is conservative rather than flattering.

## Correctness

`cuda-int8` is byte-for-byte identical to `scalar-int8` across all 22 layer
dumps, 9,420,800 elements with zero differing, and reproduces mAP and both
per-class APs exactly. `scalar-int8` was already verified identical to
`vnni-int8`, so the GPU kernel matches the AVX-VNNI kernel. This is provable
rather than approximate because int32 accumulation is exact and
order-independent, and every floating-point operation stays on the host.

`cuda-int8-fused` cannot make that claim and is not meant to. CUDA's `expf` is
not glibc's, and one unit in the last place is enough to flip `lround` where a
value sits on a requantization tie. It shows up as at most 3 INT8 codes on 3 of
the 22 layers, with no bias, and it moves no detection across a threshold: mAP
and both per-class APs are unchanged. The two paths are kept separate so the
exact result stays available.

`quantization/compare_dumps.py` is the gate for all of this and exits nonzero.

## What INT8 actually did to latency

Pairing each runtime against its own FP32 measurement, which is the only way to
attribute a change to precision rather than to a change of runtime.

| Runtime | FP32 | INT8 | Effect |
|---------|------|------|--------|
| ONNX Runtime | 24.1 ms | 30.8 ms | 1.28x slower |
| OpenVINO | 28.5 ms | 13.9 ms | 2.05x faster |
| Custom engine, scalar loop | 3105.9 ms | 3578.7 ms | 1.15x slower |
| Custom engine, AVX-VNNI | 3105.9 ms | 233.5 ms | 13.3x faster |

Both engine rows share the same FP32 baseline because they are the same engine
with the same weights, differing only in how the multiply-accumulates are
issued.

INT8 is not intrinsically faster to compute. It is faster only when the kernel
issues an instruction that consumes more 8-bit lanes per cycle than the FP32
equivalent. OpenVINO fuses convolution, bias and activation into VNNI kernels
and collects that. ONNX Runtime here quantizes only Conv, so the graph converts
format between nearly every layer and the conversions cost more than the faster
convolutions save. The engine's scalar loop issues one multiply at a time and
collects nothing, which is why quantizing it alone made it slightly slower.

There is no TensorRT precision pair. The 4.9 ms INT8 figure is measured against
PyTorch FP32 at 8.0 ms, so that ratio mixes a runtime change with a precision
change and is not comparable to the rows above.

## Remaining gap

OpenVINO INT8 at 13.9 ms is still 16.8x faster than the vectorized engine at
233.5 ms. The engine runs on one core against eight, which accounts for most of
it. The rest is cache blocking and fusing the activation into the convolution.
