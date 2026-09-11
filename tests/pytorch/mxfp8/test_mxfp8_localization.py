# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for experimental per-locality-domain MXFP8 quantization."""

import os

import pytest
import torch

import transformer_engine.pytorch as te


def _benchmark_ms(function, warmup: int = 20, iterations: int = 100) -> float:
    """Measure average CUDA execution time, including joined side-stream work."""
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


def _localization_available() -> bool:
    try:
        from torch.cuda.green_contexts import is_localization_supported
        from torch.cuda.memory import get_num_locality_domains
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    try:
        supported = is_localization_supported(device)
    except TypeError:
        supported = is_localization_supported()
    return supported and get_num_locality_domains(device) == 2


@pytest.mark.skipif(
    not _localization_available(), reason="CUDA localization is unavailable"
)
def test_mxfp8_rowwise_localized_pair() -> None:
    """Localized halves must match a full-tensor rowwise quantization."""
    tensor = torch.randn((256, 128), dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    quantizer.optimize_for_gemm = True

    reference = quantizer(tensor)
    localized = te.localize_mxfp8_tensor(tensor, quantizer)
    outputs = localized.quantize()
    torch.cuda.synchronize()

    assert len(outputs) == 2
    assert tuple(outputs[0].shape) == (128, 128)
    assert tuple(outputs[1].shape) == (128, 128)
    torch.testing.assert_close(
        localized.dequantize(),
        reference.dequantize(),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(
    not _localization_available(), reason="CUDA localization is unavailable"
)
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
def test_mxfp8_rowwise_localized_performance() -> None:
    """Compare full-chip and two-domain quantization for [4096, 32768]."""
    shape = (4096, 32768)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    quantizer.optimize_for_gemm = True

    baseline_output = quantizer.make_empty(
        shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    localized = te.localize_mxfp8_tensor(tensor, quantizer)

    baseline_ms = _benchmark_ms(
        lambda: quantizer.update_quantized(tensor, baseline_output)
    )
    localized_ms = _benchmark_ms(localized.quantize)

    assert baseline_ms > 0.0
    assert localized_ms > 0.0
    print(
        f"\nMXFP8 localization {shape}: baseline={baseline_ms:.3f} ms, "
        f"localized={localized_ms:.3f} ms, speedup={baseline_ms / localized_ms:.3f}x"
    )
