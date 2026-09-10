# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for experimental per-locality-domain MXFP8 quantization."""

import pytest
import torch

import transformer_engine.pytorch as te


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
