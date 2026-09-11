# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Experimental per-locality-domain MXFP8 quantization."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor


_LOCALIZATION_CONTEXTS = {}


def _get_localization_context(device_index: int):
    context = _LOCALIZATION_CONTEXTS.get(device_index)
    if context is not None:
        return context

    try:
        from torch.cuda.green_contexts import GreenContext, is_localization_supported
        from torch.cuda.memory import LocalizedMemPool, get_num_locality_domains
    except ImportError as exc:
        raise RuntimeError(
            "This PyTorch build does not provide CUDA locality-domain APIs"
        ) from exc

    try:
        supported = is_localization_supported(device_index)
    except TypeError:
        supported = is_localization_supported()
    if not supported:
        raise RuntimeError(f"CUDA device {device_index} does not support localization")

    num_domains = get_num_locality_domains(device_index)
    if num_domains != 2:
        raise RuntimeError(f"Expected exactly 2 locality domains, got {num_domains}")

    green_contexts = tuple(
        GreenContext.create(locality_domain_id=domain, device_id=device_index)
        for domain in range(2)
    )
    mempools = tuple(
        LocalizedMemPool(domain, device=device_index) for domain in range(2)
    )
    for pool in mempools:
        pool.alloc_in_order = True
    streams = tuple(green_context.Stream() for green_context in green_contexts)
    context = (green_contexts, mempools, streams)
    _LOCALIZATION_CONTEXTS[device_index] = context
    return context


class MXFP8LocalizedPair:
    """Two independently allocated MXFP8 tensors, one per GPU locality domain.

    Input rows are copied once into persistent, locality-domain-backed tensors.
    Calls to :meth:`quantize` time only the quantization work; refreshing the
    localized inputs is a separate operation. The fork/join is CUDA-graph
    capturable when capture starts on the parent stream.
    """

    def __init__(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        outputs: Tuple[MXFP8Tensor, MXFP8Tensor],
        quantizer: MXFP8Quantizer,
        green_contexts: Tuple[object, object],
        mempools: Tuple[object, object],
        streams: Tuple[torch.cuda.Stream, torch.cuda.Stream],
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.quantizer = quantizer
        self.green_contexts = green_contexts
        self.mempools = mempools
        self.streams = streams
        self._fork_event = torch.cuda.Event(enable_timing=False)
        self._join_events = (
            torch.cuda.Event(enable_timing=False),
            torch.cuda.Event(enable_timing=False),
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        quantizer: MXFP8Quantizer,
    ) -> "MXFP8LocalizedPair":
        """Allocate localized input and output halves and copy ``tensor`` into them."""
        if tensor.device.type != "cuda":
            raise ValueError("MXFP8 localization requires a CUDA tensor")
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tuple(tensor.shape)}")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "The specialized MXFP8 cast-only kernel requires FP16 or BF16 input "
                f"(got {tensor.dtype})"
            )
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if not quantizer.rowwise_usage:
            raise ValueError(
                "MXFP8 localization requires rowwise output; columnwise-only is unsupported"
            )
        if quantizer.with_2d_quantization:
            raise ValueError("MXFP8 localization does not support 2D quantization")
        if quantizer.internal:
            raise ValueError(
                "MXFP8 localization currently requires quantizer.internal=False"
            )

        rows, cols = tensor.shape
        if rows % 2 != 0:
            raise ValueError(f"Row count must be divisible by 2, got {rows}")
        rows_per_domain = rows // 2
        if rows_per_domain % 32 != 0 or cols % 32 != 0:
            raise ValueError(
                "Each input half must satisfy MXFP8 shape alignment "
                f"(got half shape {(rows_per_domain, cols)})"
            )
        if not quantizer.columnwise_usage and cols % 128 != 0:
            raise ValueError(
                "The specialized rowwise-only kernel requires 128-aligned columns "
                f"(got {cols})"
            )
        if quantizer.optimize_for_gemm and rows_per_domain % 128 != 0:
            raise ValueError(
                "The cast+swizzle path requires each row half to be 128-aligned "
                f"(got half shape {(rows_per_domain, cols)})"
            )

        device_index = tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        green_contexts, mempools, streams = _get_localization_context(device_index)

        parent_stream = torch.cuda.current_stream(device_index)
        fork_event = torch.cuda.Event(enable_timing=False)
        fork_event.record(parent_stream)
        join_events = []
        inputs = []
        outputs = []

        for domain in range(2):
            start = domain * rows_per_domain
            end = start + rows_per_domain
            stream = streams[domain]
            stream.wait_event(fork_event)
            tensor.record_stream(stream)
            # Eager LocalizedMemPool callbacks expect pool outermost and the
            # matching green stream innermost.
            with torch.cuda.use_mem_pool(mempools[domain]):
                with torch.cuda.stream(stream):
                    local_input = torch.empty(
                        (rows_per_domain, cols),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    local_input.copy_(tensor[start:end])
                    local_output = quantizer.make_empty(
                        local_input.shape,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
            done = torch.cuda.Event(enable_timing=False)
            done.record(stream)
            join_events.append(done)
            inputs.append(local_input)
            outputs.append(local_output)

        for event in join_events:
            parent_stream.wait_event(event)

        return cls(
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            quantizer=quantizer,
            green_contexts=green_contexts,
            mempools=mempools,
            streams=streams,
        )

    def copy_from(
        self,
        tensor: torch.Tensor,
        parent_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Refresh the persistent localized input halves from a full tensor."""
        if tensor.device != self.inputs[0].device:
            raise ValueError(
                f"Input device {tensor.device} does not match localized device "
                f"{self.inputs[0].device}"
            )
        expected_shape = (self.inputs[0].shape[0] * 2, self.inputs[0].shape[1])
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected input shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if parent_stream is None:
            parent_stream = torch.cuda.current_stream(tensor.device)

        self._fork_event.record(parent_stream)
        rows_per_domain = self.inputs[0].shape[0]
        for domain, (local_input, stream) in enumerate(zip(self.inputs, self.streams)):
            stream.wait_event(self._fork_event)
            tensor.record_stream(stream)
            with torch.cuda.stream(stream):
                start = domain * rows_per_domain
                local_input.copy_(tensor[start : start + rows_per_domain])
            self._join_events[domain].record(stream)
        for event in self._join_events:
            parent_stream.wait_event(event)

    def quantize(
        self,
        parent_stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[MXFP8Tensor, MXFP8Tensor]:
        """Quantize both localized halves concurrently with a fork/join."""
        if parent_stream is None:
            parent_stream = torch.cuda.current_stream(self.inputs[0].device)

        # Inputs are persistent and already populated. Record the fork here so
        # any preceding copy_from work on the parent stream is also ordered.
        self._fork_event.record(parent_stream)
        for domain, (local_input, local_output, stream) in enumerate(
            zip(self.inputs, self.outputs, self.streams)
        ):
            stream.wait_event(self._fork_event)
            with torch.cuda.stream(stream):
                self.quantizer.update_quantized(local_input, local_output)
            self._join_events[domain].record(stream)
        for event in self._join_events:
            parent_stream.wait_event(event)
        return self.outputs

    def dequantize(self) -> torch.Tensor:
        """Dequantize and concatenate both row partitions."""
        return torch.cat(tuple(output.dequantize() for output in self.outputs), dim=0)


def localize_mxfp8_tensor(
    tensor: torch.Tensor,
    quantizer: MXFP8Quantizer,
) -> MXFP8LocalizedPair:
    """Construct an eager, two-domain localized MXFP8 quantization pair."""
    return MXFP8LocalizedPair.from_tensor(tensor, quantizer)
