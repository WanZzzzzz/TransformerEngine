VMM output localization prototype
=================================

This prototype evaluates a focused MLPerf DSV3 pattern:

``ordinary activation -> MXFP8 quantization -> small GEMM``

It does not localize LayerNorm. Profiles do not show a reliable direct
LayerNorm-to-quantization producer relationship, especially in backward where
gradient accumulation or other elementwise work can occur between the two.

Workflow
--------

The activation remains an ordinary PyTorch allocation. The quantized MXFP8
data buffers use one contiguous virtual address range whose first and second
row partitions are backed by physical memory in locality domains 0 and 1.

Two green-context streams quantize the partitions concurrently:

1. The parent stream records a fork event after the input is ready.
2. Each green stream waits for the fork event.
3. Each stream reads its ordinary-memory input partition and writes its
   locality-backed output partition.
4. The parent stream waits for both completion events.
5. Either an unchanged full-chip GEMM consumes the single MXFP8 tensor, or
   each green stream immediately runs GEMM on its local activation partition.
6. The parent stream joins both partitioned GEMMs before consuming the ordinary
   contiguous output tensor.

There is no explicit copy. Input reads by quantization are not localized. In
the partitioned-GEMM variant, activation reads and independent cuBLAS
workspaces are localized, while weight reads and GEMM output writes use
ordinary allocations.

Producer-output experiment
--------------------------

For a known producer with an ``out=`` interface, allocate the workspace input
up front and write the producer result directly into it:

.. code-block:: python

   workspace = te.MXFP8VMMWorkspace.empty(
       x.shape,
       dtype=x.dtype,
       device=x.device,
       quantizer=quantizer,
   )
   torch.add(x, residual, out=workspace.input)
   workspace.quantize()

This avoids an explicit localization copy. It is a focused benchmark interface:
PyTorch operations with ``out=`` do not participate in ordinary autograd
recording. Megatron's fused MLA rotary-KV function likewise accepts optional
persistent ``out_key`` and ``out_value`` tensors, which may be VMM allocations.
The rotary-Q kernel is in-place and therefore inherits the q-up GEMM output
allocation.

Example
-------

.. code-block:: python

   import torch
   import transformer_engine.pytorch as te
   from transformer_engine.pytorch.cpp_extensions import general_gemm

   x = torch.randn((4096, 32768), device="cuda", dtype=torch.bfloat16)
   quantizer = te.MXFP8Quantizer(
       fp8_dtype=te.DType.kFloat8E4M3,
       rowwise=True,
       columnwise=True,
   )
   quantizer.optimize_for_gemm = True

   workspace = te.localize_mxfp8_output_vmm(x, quantizer)
   workspace.quantize()

   # workspace.output remains one GEMM-ready MXFP8Tensor.
   general_gemm(
       quantized_weight,
       workspace.output,
       out_dtype=torch.bfloat16,
       out=output,
   )

   # Or keep quantization and its row-partition GEMM on each green stream.
   workspace.quantize_and_gemm(quantized_weight, output)

CUDA Graph capture
------------------

Capture quantization and GEMM together. Graph replay removes Python and TE
dispatch gaps between the two green launches and keeps the join-to-GEMM
dependency on the GPU.

Focused validation
------------------

.. code-block:: bash

   pytest -q tests/pytorch/mxfp8/test_mxfp8_localization.py \
     -k bidirectional_swizzled_vmm

   RUN_BENCHMARK_TESTS=1 \
   MXFP8_LOCALIZATION_USE_CUDA_GRAPH=1 \
   MXFP8_LOCALIZATION_GEMM_N=256 \
   pytest -q -s tests/pytorch/mxfp8/test_mxfp8_localization.py \
     -k bidirectional_swizzled_vmm_performance

   RUN_BENCHMARK_TESTS=1 \
   MXFP8_LOCALIZATION_USE_CUDA_GRAPH=1 \
   pytest -q -s tests/pytorch/mxfp8/test_mxfp8_localization.py \
     -k vmm_add_producer_performance

Prototype limitations
---------------------

* Exactly two locality domains and equal, aligned row partitions.
* Bidirectional MXFP8 with fused GEMM scale swizzling only.
* Ordinary quantization input and GEMM output allocations.
* GEMM is split across output rows; activation inputs and cuBLAS workspaces are
  localized, but the shared weight is not.
* Scale buffers remain ordinary allocations.
* Explicit workspace lifetime management is required.
