# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LazyMemoryAllocator's deferred ("lazy") pinning contract.

LazyMemoryAllocator must not create a CUDA context when it is constructed: host
pinning (``cudaHostRegister``) creates a primary context on the current device,
so pinning in ``__init__`` would squat a context on the default device
(``cuda:0``) at multiprocess-server start, before any worker connects.

Pinning runs in a background thread triggered by ``ensure_pinning`` (normally
from the worker-registration path). Nothing may assume the pool is ready
before pinning finishes: the usable address space stays at zero until the init
chunk is pinned, so early allocations return ``None`` (a cache miss) instead
of blocking or receiving unpinned memory.

These tests verify the documented contract by observing CUDA side effects (the
pin/unpin calls and the device context they run in) and the public
``pin_state`` machine rather than by inspecting private state, so they need no
GPU.
"""

# Standard
from unittest.mock import call, patch
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator, PinState

PIN_CHUNK = LazyMemoryAllocator.PIN_CHUNK_SIZE


@pytest.fixture
def mock_torch_dev():
    """Patch the CUDA device layer; a MagicMock auto-supports ``with dev``."""
    with patch("lmcache.v1.lazy_memory_allocator.torch_dev") as td:
        td.ext.is_pin_supported = True
        td.ext.pin_memory.return_value = True
        td.is_available.return_value = True
        td.current_device.return_value = 0
        yield td


@pytest.fixture
def mock_tma():
    """Patch TensorMemoryAllocator to isolate the lazy logic from it."""
    with patch("lmcache.v1.lazy_memory_allocator.TensorMemoryAllocator") as tma:
        yield tma


def _make_allocator(
    init: int = PIN_CHUNK, final: int = PIN_CHUNK
) -> LazyMemoryAllocator:
    """Build an allocator. init == final keeps expansion a no-op."""
    return LazyMemoryAllocator(init, final)


def _wait_state(
    allocator: LazyMemoryAllocator, state: PinState, timeout: float = 5.0
) -> bool:
    """Poll until the allocator reaches ``state`` or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if allocator.pin_state is state:
            return True
        time.sleep(0.005)
    return allocator.pin_state is state


def _wait_not_pinning(allocator: LazyMemoryAllocator, timeout: float = 5.0) -> bool:
    """Poll until the allocator leaves PINNING or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if allocator.pin_state is not PinState.PINNING:
            return True
        time.sleep(0.005)
    return allocator.pin_state is not PinState.PINNING


def test_construction_does_not_touch_cuda(mock_torch_dev, mock_tma):
    """The core contract: __init__ neither pins nor queries the device."""
    allocator = _make_allocator()

    mock_torch_dev.ext.pin_memory.assert_not_called()
    mock_torch_dev.current_device.assert_not_called()
    assert allocator.pin_state is PinState.NOT_STARTED


def test_construction_exposes_zero_address_space(mock_torch_dev, mock_tma):
    """Until pinning succeeds, the underlying pool must not be allocatable."""
    _make_allocator()

    _, kwargs = mock_tma.call_args
    assert kwargs["init_address_space"] == 0


def test_ensure_pinning_pins_initial_chunk_on_device(mock_torch_dev, mock_tma):
    """ensure_pinning(d) pins the initial chunk inside device d's context."""
    allocator = _make_allocator()

    allocator.ensure_pinning(3)

    assert _wait_state(allocator, PinState.READY)
    mock_torch_dev.ext.pin_memory.assert_called_once()
    mock_torch_dev.device.assert_called_with(3)

    allocator.close()


def test_ensure_pinning_does_not_block_the_caller(mock_torch_dev, mock_tma):
    """ensure_pinning returns immediately while the pin runs in background."""
    release = threading.Event()

    def gated_pin(ptr, size, flag):
        release.wait(timeout=5)
        return True

    mock_torch_dev.ext.pin_memory.side_effect = gated_pin
    allocator = _make_allocator()

    start = time.monotonic()
    allocator.ensure_pinning(3)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, "ensure_pinning must not wait for the pin itself"
    assert allocator.pin_state is PinState.PINNING

    release.set()
    assert _wait_state(allocator, PinState.READY)
    allocator.close()


def test_ensure_pinning_is_idempotent(mock_torch_dev, mock_tma):
    """A second ensure_pinning does not re-pin or rebind the device."""
    allocator = _make_allocator()

    allocator.ensure_pinning(3)
    assert _wait_state(allocator, PinState.READY)
    pins_after_first = mock_torch_dev.ext.pin_memory.call_count

    allocator.ensure_pinning(5)

    assert mock_torch_dev.ext.pin_memory.call_count == pins_after_first
    assert call(5) not in mock_torch_dev.device.call_args_list

    allocator.close()


def test_allocate_fallback_pins_synchronously_on_current_device(
    mock_torch_dev, mock_tma
):
    """The standalone fallback: with no registration trigger, the first
    allocate pins on the current device and waits until the pool is READY
    (preserving the previous direct-use semantics)."""
    mock_torch_dev.current_device.return_value = 2
    allocator = _make_allocator()

    mock_torch_dev.ext.pin_memory.assert_not_called()

    allocator.allocate(torch.Size([16]), torch.uint8)

    assert allocator.pin_state is PinState.READY
    mock_tma.return_value.allocate.assert_called_once()
    mock_torch_dev.ext.pin_memory.assert_called_once()
    mock_torch_dev.current_device.assert_called()
    mock_torch_dev.device.assert_called_with(2)

    allocator.close()


def test_second_allocate_does_not_repin(mock_torch_dev, mock_tma):
    """Once pinned, later allocations do not pin again."""
    allocator = _make_allocator()

    allocator.allocate(torch.Size([16]), torch.uint8)
    pins_after_first = mock_torch_dev.ext.pin_memory.call_count

    allocator.allocate(torch.Size([16]), torch.uint8)

    assert mock_torch_dev.ext.pin_memory.call_count == pins_after_first

    allocator.close()


def test_allocate_returns_none_until_ready_then_succeeds(mock_torch_dev):
    """No readiness assumption: while the pin is in flight, allocate fails
    fast with None (a cache miss) instead of blocking or handing out
    unpinned memory; after READY the same allocation succeeds."""
    release = threading.Event()

    def gated_pin(ptr, size, flag):
        release.wait(timeout=5)
        return True

    mock_torch_dev.ext.pin_memory.side_effect = gated_pin
    allocator = _make_allocator()

    allocator.ensure_pinning(1)
    assert allocator.pin_state is PinState.PINNING

    start = time.monotonic()
    obj = allocator.allocate(torch.Size([16]), torch.uint8)
    elapsed = time.monotonic() - start

    assert obj is None, "allocation before READY must fail as a cache miss"
    assert elapsed < 1.0, "allocation before READY must not block on the pin"

    release.set()
    assert _wait_state(allocator, PinState.READY)

    obj = allocator.allocate(torch.Size([16]), torch.uint8)
    assert obj is not None

    allocator.close()


def test_wait_until_ready_reports_success(mock_torch_dev, mock_tma):
    """wait_until_ready blocks until the pool is READY and returns True."""
    allocator = _make_allocator()

    allocator.ensure_pinning(0)

    assert allocator.wait_until_ready(timeout=5) is True
    assert allocator.pin_state is PinState.READY

    allocator.close()


def test_expansion_grows_pool_to_final_size(mock_torch_dev):
    """After READY, background expansion keeps pinning up to final_size."""
    allocator = LazyMemoryAllocator(PIN_CHUNK, 3 * PIN_CHUNK)

    allocator.ensure_pinning(0)
    assert allocator.wait_until_ready(timeout=5)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if allocator.pinned_bytes >= 3 * PIN_CHUNK:
            break
        time.sleep(0.005)

    assert allocator.pinned_bytes == 3 * PIN_CHUNK
    assert mock_torch_dev.ext.pin_memory.call_count == 3

    allocator.close()


def test_expansion_failure_keeps_pool_ready_at_committed_size(mock_torch_dev):
    """If an expansion chunk fails to pin, the pool stays READY at its
    committed size; the address space never runs ahead of the pinned
    bytes."""
    # Init chunk pins fine; the first expansion chunk fails.
    mock_torch_dev.ext.pin_memory.side_effect = [True, False]
    allocator = LazyMemoryAllocator(PIN_CHUNK, 3 * PIN_CHUNK)

    allocator.ensure_pinning(0)
    assert allocator.wait_until_ready(timeout=5)

    # Give the expansion thread time to hit the failure and stop.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if mock_torch_dev.ext.pin_memory.call_count >= 2:
            break
        time.sleep(0.005)

    assert allocator.pin_state is PinState.READY
    assert allocator.pinned_bytes == PIN_CHUNK

    allocator.close()


def test_wait_until_ready_reports_failure(mock_torch_dev):
    """wait_until_ready returns False when the pin attempt fails."""
    mock_torch_dev.ext.pin_memory.return_value = False
    allocator = _make_allocator()

    allocator.ensure_pinning(0)

    assert allocator.wait_until_ready(timeout=5) is False
    assert allocator.pin_state is not PinState.READY

    allocator.close()


def test_pin_failure_retries_then_fails_permanently(mock_torch_dev):
    """Init-pin failures retry per trigger, then latch FAILED; the pool
    stays disabled (None allocations) but nothing crashes."""
    mock_torch_dev.ext.pin_memory.return_value = False
    allocator = _make_allocator()

    for _ in range(LazyMemoryAllocator.MAX_PIN_ATTEMPTS):
        allocator.ensure_pinning(0)
        assert _wait_not_pinning(allocator)

    assert allocator.pin_state is PinState.FAILED

    # Further triggers are no-ops and allocations keep failing cleanly
    pins_so_far = mock_torch_dev.ext.pin_memory.call_count
    allocator.ensure_pinning(0)
    assert allocator.pin_state is PinState.FAILED
    assert mock_torch_dev.ext.pin_memory.call_count == pins_so_far
    assert allocator.allocate(torch.Size([16]), torch.uint8) is None

    allocator.close()


def test_close_before_pinning_is_safe(mock_torch_dev, mock_tma):
    """close() before any pinning neither raises nor unpins."""
    allocator = _make_allocator()

    allocator.close()

    mock_torch_dev.ext.unpin_memory.assert_not_called()


def test_ensure_pinning_after_close_is_noop(mock_torch_dev, mock_tma):
    """ensure_pinning after close() must not pin (close/first-use race guard)."""
    allocator = _make_allocator()
    allocator.close()

    allocator.ensure_pinning(3)

    assert allocator.pin_state is PinState.NOT_STARTED
    time.sleep(0.05)
    mock_torch_dev.ext.pin_memory.assert_not_called()


def test_close_after_pinning_unpins(mock_torch_dev, mock_tma):
    """close() after pinning unpins the chunk it recorded."""
    allocator = _make_allocator()
    allocator.ensure_pinning(3)
    assert _wait_state(allocator, PinState.READY)

    allocator.close()

    mock_torch_dev.ext.unpin_memory.assert_called_once()


def test_close_during_pinning_is_responsive(mock_torch_dev):
    """close() while a multi-chunk pin is in flight stops at the next chunk
    boundary instead of pinning the whole pool."""
    first_pin_entered = threading.Event()
    release = threading.Event()

    def gated_pin(ptr, size, flag):
        first_pin_entered.set()
        release.wait(timeout=5)
        return True

    mock_torch_dev.ext.pin_memory.side_effect = gated_pin
    allocator = LazyMemoryAllocator(4 * PIN_CHUNK, 4 * PIN_CHUNK)

    allocator.ensure_pinning(0)
    assert first_pin_entered.wait(timeout=5)

    closer = threading.Thread(target=allocator.close)
    closer.start()
    release.set()
    closer.join(timeout=5)

    assert not closer.is_alive(), "close() must not wait for the full pool"
    assert mock_torch_dev.ext.pin_memory.call_count < 4


def test_concurrent_ensure_pinning_pins_once(mock_torch_dev, mock_tma):
    """Under concurrent ensure_pinning calls, exactly one thread pins."""
    allocator = _make_allocator()
    start = threading.Barrier(8)

    def worker(device):
        start.wait()  # maximize contention on the state lock
        allocator.ensure_pinning(device)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert _wait_state(allocator, PinState.READY)
    mock_torch_dev.ext.pin_memory.assert_called_once()

    allocator.close()
