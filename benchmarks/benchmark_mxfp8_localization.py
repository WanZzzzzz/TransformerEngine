# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark rowwise MXFP8 quantization on two localized GPU partitions."""

import argparse

import torch

import transformer_engine.pytorch as te


def _time_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    """Run correctness validation and compare full-chip and localized casts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=16384)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--include-input-copy",
        action="store_true",
        help="Time copying the full input into the two localized input tensors",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tensor = torch.randn((args.rows, args.cols), dtype=dtype, device="cuda")

    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    quantizer.optimize_for_gemm = True

    baseline_output = quantizer.make_empty(
        tensor.shape, dtype=dtype, device=tensor.device
    )
    localized = te.localize_mxfp8_tensor(tensor, quantizer)

    def baseline_quantize() -> None:
        quantizer.update_quantized(tensor, baseline_output)

    def localized_quantize() -> None:
        if args.include_input_copy:
            localized.copy_from(tensor)
        localized.quantize()

    baseline_quantize()
    localized.quantize()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        localized.dequantize(),
        baseline_output.dequantize(),
        atol=0.0,
        rtol=0.0,
    )

    baseline_ms = _time_ms(baseline_quantize, args.warmup, args.iterations)
    localized_ms = _time_ms(localized_quantize, args.warmup, args.iterations)

    # Approximate traffic: high-precision input + FP8 output + one E8M0
    # scale per 32 input elements. This excludes allocator and event traffic.
    traffic_bytes = tensor.numel() * (tensor.element_size() + 1 + 1 / 32)
    baseline_gbps = traffic_bytes / (baseline_ms * 1e-3) / 1e9
    localized_gbps = traffic_bytes / (localized_ms * 1e-3) / 1e9

    print(
        f"shape={tuple(tensor.shape)} dtype={dtype} copy_in_timing={args.include_input_copy}"
    )
    print(f"baseline:  {baseline_ms:.3f} ms  {baseline_gbps:.1f} GB/s")
    print(f"localized: {localized_ms:.3f} ms  {localized_gbps:.1f} GB/s")
    print(f"speedup:   {baseline_ms / localized_ms:.3f}x")


if __name__ == "__main__":
    main()
