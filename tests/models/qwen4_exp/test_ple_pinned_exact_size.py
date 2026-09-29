# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The offloaded PLE table must be pinned at its exact size, not rounded up.

pin_memory=True goes through PyTorch's caching host allocator, which rounds each
request up to the next power of two (a 23.9 GiB table pins 32 GiB).
"""

import gc

import pytest
import regex as re
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.cuda.is_available()),
    reason="exact-size pinning is CUDA-only",
)

# Just over 64 MiB: a power-of-two allocator would back it with 128 MiB.
ROWS, DIM = 16_400, 4096


def _mapping_bytes(ptr: int) -> int:
    with open("/proc/self/maps") as f:
        maps = f.read()
    for lo, hi in re.findall(r"^([0-9a-f]+)-([0-9a-f]+) ", maps, re.M):
        if int(lo, 16) <= ptr < int(hi, 16):
            return int(hi, 16) - int(lo, 16)
    raise AssertionError("no mapping contains the tensor")


def _allocate(rows: int, dim: int) -> torch.Tensor:
    from vllm.models.qwen4_exp.common.ngram_embedding import (
        Qwen4ExpPLEPinnedHostEmbedding as cls,
    )

    # allocate_embedding_weight uses no layer state, so skip __init__.
    layer = cls.__new__(cls)
    return layer.allocate_embedding_weight(rows, dim, torch.float8_e4m3fn)


def test_pinned_ple_table_is_exact_size():
    w = _allocate(ROWS, DIM)
    assert w.shape == (ROWS, DIM)
    assert w.is_pinned()
    assert _mapping_bytes(w.data_ptr()) < ROWS * DIM * 1.01


def test_reallocate_after_free():
    # Freed tables must be unregistered, or a later allocation reusing the
    # address range fails with cudaErrorHostMemoryAlreadyRegistered.
    for i in range(4):
        w = _allocate(ROWS, DIM)
        assert w.is_pinned()
        assert _mapping_bytes(w.data_ptr()) < ROWS * DIM * 1.01
        w.view(torch.uint8).fill_(i)
        del w
        gc.collect()


def test_uva_view_reads_registered_table():
    import vllm._C_stable_libtorch  # noqa: F401  registers torch.ops._C

    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    rows, dim = 4096, 2560
    w = _allocate(rows, dim)
    src = torch.randint(0, 255, (rows, dim), dtype=torch.uint8)
    w.view(torch.uint8).copy_(src)
    uva = get_accelerator_view_from_cpu_tensor(w)
    idx = torch.randint(0, rows, (1024,), device=uva.device)
    assert torch.equal(uva.view(torch.uint8)[idx].cpu(), src[idx.cpu()])


def test_empty_table():
    assert _allocate(0, DIM).shape == (0, DIM)
