# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from typing import List, Optional, Union
import ctypes
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryAllocator,
)
from lmcache.v1.system_detection import NUMAMapping
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# Helper functions
def get_numa_id(numa_mapping: NUMAMapping) -> int:
    """
    Get the NUMA ID for the current GPU

    Args:
        numa_mapping (NUMAMapping): The NUMA mapping object.

    Returns:
        int: The NUMA ID for the current GPU.

    Raises:
        KeyError: If GPU id is not detected in the numa mapping.
    """
    gpu_id = torch_dev.current_device() if torch_dev.is_available() else 0
    return numa_mapping.gpu_to_numa_mapping[gpu_id]


def align_to(size: int, align_size: int) -> int:
    """
    Align the given size to the nearest multiple of align_size.

    Args:
        size (int): The size to align.
        align_size (int): The alignment size, MUST BE a power of two.

    Returns:
        int: The aligned size.
    """
    return (size + align_size - 1) & (~(align_size - 1))


class PinState(Enum):
    """Lifecycle states of the deferred host-pinning state machine.

    NOT_STARTED: No pinning has been triggered, or a failed attempt is
        awaiting retry. The pool exposes zero usable address space.
    PINNING: A background thread is pinning the init chunk. The pool
        still exposes zero usable address space.
    READY: The init chunk is pinned and committed; allocations succeed.
        Background expansion may still be growing the pool.
    FAILED: Pinning failed ``MAX_PIN_ATTEMPTS`` times. The pool
        permanently exposes zero usable address space (every allocation
        returns ``None``) but the process stays alive.
    """

    NOT_STARTED = auto()
    PINNING = auto()
    READY = auto()
    FAILED = auto()


# Main class
class LazyMemoryAllocator(MemoryAllocatorInterface):
    """
    Allocates CPU (numa) pinned memory with a initial size and expand
    the size to the required size in the background.

    Background expansion logic:
    - After registering X GB memory, we call sbrk and updates _curr_size
    - Once everything is registered, the background thread stops

    Deferred pinning:
    - Pinning (``cudaHostRegister``) creates a CUDA context on the current
      device, so doing it in ``__init__`` would squat a context on ``cuda:0``
      at server start. Instead, :meth:`ensure_pinning` (called from the
      worker-registration path, or lazily by the first :meth:`allocate`)
      starts a background thread that pins the init chunk and then keeps
      expanding to the final size.

    No-readiness-assumption invariant:
    - The usable address space always trails the pinned bytes: it starts at
      zero and grows (``sbrk``) only after each pin succeeds, so an
      allocation can never hand out unpinned memory. Once pinning is in
      flight, allocations simply return ``None`` until the init chunk is
      committed (callers already treat allocation failure as a cache miss);
      nothing on the request path blocks waiting for the pin, and nothing
      has to assume the pool is ready before it actually is.
      :attr:`pin_state` exposes the state explicitly and
      :meth:`wait_until_ready` lets direct users synchronize when they
      need a usable pool deterministically.
    - Standalone fallback: if nothing ever triggered pinning (no
      registration path, e.g. tools and tests using the allocator
      directly), the first :meth:`allocate` pins synchronously on the
      current device, preserving the previous direct-use semantics.
    """

    PIN_CHUNK_SIZE = 1 << 26  # 64 MB pin chunk
    COMMIT_SIZE = 1 << 30  # Do a commit every 1 GB
    LOG_INTERVAL = 10 << 30  # Log expansion progress every 10 GB
    MAX_PIN_ATTEMPTS = 3  # Init-chunk pin retries before giving up

    def __init__(
        self,
        init_size: int,
        final_size: int,
        align_bytes: int = AddressManager.ALIGN_BYTES,
        numa_mapping: NUMAMapping | None = None,
    ):
        """
        Args:
            init_size (int): Initial size of the memory allocation in bytes.
            final_size (int): Final size of the memory allocation in bytes.
            align_bytes (int, optional): Alignment in for the underlying allocations
            numa_mapping (NUMAMapping | None, optional): NUMA mapping used to bind
                the host buffer to a NUMA node. ``None`` disables NUMA binding.

        Raises:
            RuntimeError: If the active device backend does not support memory
                pinning.
        """
        # Whether using NUMA allocation
        self._use_numa = numa_mapping is not None
        # Final size of the allocation
        self._final_size = align_to(final_size, self.PIN_CHUNK_SIZE)
        # Target size of the init chunk; the pool becomes READY once this
        # much is pinned and committed
        self._init_size = min(
            align_to(init_size, self.PIN_CHUNK_SIZE), self._final_size
        )
        # Bytes pinned so far, only mutated by the pin worker thread
        self._curr_size = 0
        # Bytes committed to the address manager (sbrk'ed) so far, only
        # mutated by the pin worker thread. Invariant: never exceeds the
        # pinned bytes, so unpinned memory is never allocatable.
        self._committed_size = 0
        # Underlying buffer for the memory allocation
        self._buffer: torch.Tensor
        if not torch_dev.ext.is_pin_supported:
            raise RuntimeError(
                f"Backend '{torch_device_type}' does not support memory "
                "pinning. LazyMemoryAllocator requires pinned memory."
            )

        # List of (ptr, size) for pinned memory chunks
        self._pin_record: list[tuple[int, int]] = []

        # Detect numa mapping (host-only allocation; no CUDA context)
        if numa_mapping is not None:
            numa_id = get_numa_id(numa_mapping)
            ptr = lmc_ops.alloc_numa_ptr(self._final_size, numa_id)
            arr_type = ctypes.c_uint8 * self._final_size
            buf = arr_type.from_address(ptr)
            self._buffer = torch.frombuffer(buf, dtype=torch.uint8)
        else:
            self._buffer = torch.empty(
                self._final_size, dtype=torch.uint8, device="cpu", pin_memory=False
            )

        # Create the tensor memory allocator with ZERO usable address space.
        # The address space opens up (sbrk) only after bytes are actually
        # pinned, so an early allocation can never receive unpinned memory;
        # it just gets None (a cache miss) instead.
        self._allocator = TensorMemoryAllocator(
            tensor=self._buffer,
            align_bytes=align_bytes,
            init_address_space=0,
        )

        # Get the address manager
        # NOTE(ApostaC): this assumes the tensor memory allocator owns the address
        # manager, which creates extra coupling in the code.
        # NOTE(ApostaC): this also assumes that the behavior of the allocation is
        # completely determined by the address manager.
        self._address_manager = self._allocator.address_manager

        # Deferred-pinning state machine; see the class docstring
        self._pin_state = PinState.NOT_STARTED
        self._pin_attempts = 0
        self._pin_device: int | torch.device | None = None
        self._closed = False
        self._state_lock = threading.Lock()
        self._stop_pinning = threading.Event()
        # Set whenever a pin attempt ends (READY, FAILED, or a retryable
        # failure) and on close; cleared when an attempt starts. Lets
        # wait_until_ready() block without polling.
        self._ready_event = threading.Event()
        # The pin worker thread; (re)created by ensure_pinning() because a
        # finished thread object cannot be restarted for a retry
        self._pin_thread: Optional[threading.Thread] = None

    # Public methods
    @property
    def pin_state(self) -> PinState:
        """Current state of the pinning state machine."""
        return self._pin_state

    @property
    def pinned_bytes(self) -> int:
        """Bytes pinned AND committed so far (usable pool size)."""
        return self._committed_size

    def ensure_pinning(self, device: int | torch.device) -> None:
        """
        Start pinning the pool on ``device`` in the background.

        Non-blocking: transitions the state machine to PINNING and spawns
        a background thread that pins the init chunk, opens the address
        space, and then keeps expanding to the final size. Idempotent and
        thread-safe: only the first call binds the device and starts the
        thread; calls while PINNING/READY/FAILED (and after close) are
        no-ops. A call after a failed attempt (state back to NOT_STARTED)
        retries with the already-bound device.

        Args:
            device (int | torch.device): Device whose CUDA context the pinned
                host pool is bound to. Typically the worker's device, learned
                from the registration path.
        """
        with self._state_lock:
            if self._pin_state is not PinState.NOT_STARTED or self._closed:
                return
            if self._pin_device is None:
                self._pin_device = device
            self._pin_state = PinState.PINNING
            self._ready_event.clear()
            self._pin_thread = threading.Thread(
                target=self._pin_worker, daemon=True, name="lazy-mem-pin-thread"
            )
            self._pin_thread.start()

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Block until the current pin attempt ends, then report readiness.

        Intended for direct library users and tests that need a usable pool
        deterministically. The MP server never calls this; its request path
        treats a not-yet-ready pool as a cache miss instead of waiting.

        Args:
            timeout: Max seconds to wait. ``None`` waits until the attempt
                ends.

        Returns:
            True if the pool is READY, False otherwise (still pinning after
            the timeout, retryable failure, FAILED, or closed).
        """
        self._ready_event.wait(timeout)
        return self._pin_state is PinState.READY

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Allocate a memory object from the pinned pool.

        Once pinning has been triggered (normally by the registration
        path), this never blocks: while the pool is not READY the usable
        address space is zero, so it simply returns ``None`` and the
        caller treats it as a cache miss. Only when nothing has triggered
        pinning at all (standalone use without a registration path) does
        the first allocation pin synchronously on the current device.
        """
        self._ensure_pinned_for_use()
        obj = self._allocator.allocate(shapes, dtypes, fmt, allocator_type)
        # HACK(ApostaC): reset the parent allocator to this lazy allocator
        # There should be a cleaner way to decouple lazy allocator and
        # tensor memory allocator
        if obj is not None:
            obj.parent_allocator = self
        return obj

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """Batched version of :meth:`allocate`; same pinning semantics."""
        self._ensure_pinned_for_use()
        # HACK(ApostaC): reset the parent allocator to this lazy allocator
        # There should be a cleaner way to decouple lazy allocator and
        # tensor memory allocator
        ret = self._allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type
        )

        if ret is None:
            return ret

        for obj in ret:
            obj.parent_allocator = self
        return ret

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ):
        self._allocator.free(memory_obj, allocator_type)

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        self._allocator.batched_free(memory_objs, allocator_type, update_stats)

    def close(self):
        """Stop pinning, unpin all pinned chunks, and release the buffer.

        Responsive even while a pin is in flight: the worker checks the
        stop event between 64 MB chunks, so close returns within roughly
        one chunk's pin time instead of waiting for the whole pool.
        """
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            thread = self._pin_thread
            # Wake any wait_until_ready() callers; they will observe a
            # non-READY state and report False.
            self._ready_event.set()

        # Join outside the state lock: the worker takes the lock for its
        # state transitions, so holding it here could deadlock.
        self._stop_pinning.set()
        if thread is not None and thread.is_alive():
            thread.join()

        # Unpin in the same device context the chunks were pinned in.
        # A non-empty pin record implies _pin_device was bound.
        if self._pin_record:
            with torch_dev.device(self._pin_device):
                for ptr, size in self._pin_record:
                    torch_dev.ext.unpin_memory(ptr)
            self._pin_record.clear()

        # Free the underlying buffer if using NUMA allocation
        if self._use_numa:
            lmc_ops.free_numa_ptr(self._buffer.data_ptr(), self._final_size)
            self._use_numa = False

    def memcheck(self) -> bool:
        return self._allocator.memcheck()

    def get_underlying_buffer(self) -> torch.Tensor:
        """
        Get the underlying buffer tensor. Will be used by RDMA registrations.
        """
        return self._buffer

    def get_address_manager(self) -> AddressManager:
        """
        Get the address manager used by this allocator.
        """
        return self._address_manager

    # Helper functions
    def _ensure_pinned_for_use(self) -> None:
        """
        Standalone fallback: pin synchronously before the first allocation.

        Normally the registration path calls :meth:`ensure_pinning` with the
        worker's device long before any allocation, and allocations during
        the in-flight pin fail fast with ``None``. This fallback only covers
        callers that never register (e.g. standalone tools): it triggers the
        pin on the current device and waits for the init chunk, preserving
        the old first-allocate-pins semantics for direct library users.
        """
        if self._pin_state is PinState.NOT_STARTED:
            device = torch_dev.current_device() if torch_dev.is_available() else 0
            logger.warning(
                "LazyMemoryAllocator: pinning triggered by allocate() instead "
                "of the registration path; binding the pinned pool to the "
                "current device %s",
                device,
            )
            self.ensure_pinning(device)
            self.wait_until_ready()

    def _pin_worker(self) -> None:
        """
        Background worker: pin the init chunk, then keep expanding.

        Init-chunk failure is retried up to ``MAX_PIN_ATTEMPTS`` (each retry
        needs a new trigger); expansion failure stops growth but keeps the
        pool READY at its current size.
        """
        try:
            self._pin_init_chunk()
        except Exception:
            self._handle_init_pin_failure()
            return

        if self._stop_pinning.is_set() or self._pin_state is not PinState.READY:
            return

        try:
            self._expand_loop()
        except Exception:
            logger.error(
                "LazyMemoryAllocator: background expansion failed; the pool "
                "stays usable at %d MB (target %d MB)",
                self._committed_size >> 20,
                self._final_size >> 20,
                exc_info=True,
            )

    def _pin_init_chunk(self) -> None:
        """
        Pin the init chunk and open the address space, then flip to READY.

        Resumes from the last pinned byte on retry. Commits (sbrk) only
        after the whole init chunk is pinned, so allocations stay disabled
        until the pool is actually usable.
        """
        while self._curr_size < self._init_size:
            if self._stop_pinning.is_set():
                return
            self._pin_memory_chunk(self._curr_size, self.PIN_CHUNK_SIZE)
            self._curr_size += self.PIN_CHUNK_SIZE

        self._commit_expansion(self._curr_size - self._committed_size)
        self._committed_size = self._curr_size
        with self._state_lock:
            self._pin_state = PinState.READY
            self._ready_event.set()
        logger.info(
            "LazyMemoryAllocator: init chunk pinned (%d MB) on device %s; "
            "the L1 pool is ready",
            self._committed_size >> 20,
            self._pin_device,
        )

    def _handle_init_pin_failure(self) -> None:
        """
        Record an init-pin failure: retry via a later trigger, or give up.

        Must be called from an ``except`` block (logs the active exception).
        """
        with self._state_lock:
            self._pin_attempts += 1
            self._ready_event.set()
            if self._pin_attempts >= self.MAX_PIN_ATTEMPTS:
                self._pin_state = PinState.FAILED
                logger.error(
                    "LazyMemoryAllocator: init-chunk pinning failed %d times; "
                    "giving up. The L1 pool stays disabled (all allocations "
                    "return None) but the process keeps running",
                    self._pin_attempts,
                    exc_info=True,
                )
            else:
                self._pin_state = PinState.NOT_STARTED
                logger.error(
                    "LazyMemoryAllocator: init-chunk pinning failed "
                    "(attempt %d/%d); will retry on the next trigger",
                    self._pin_attempts,
                    self.MAX_PIN_ATTEMPTS,
                    exc_info=True,
                )

    def _pin_memory_chunk(self, offset: int, size: int):
        """
        Pin a chunk of memory on the bound device.

        Args:
            offset (int): Offset in the buffer to pin.
            size (int): Size of the memory chunk in bytes.

        Raises:
            ValueError: If offset/size are not chunk-aligned or exceed the
                buffer.
            RuntimeError: If the underlying pin call fails. The pool maps
                pinned memory into the GPU address space (zero-copy), so an
                unpinned chunk must never be exposed to allocations.
        """
        if offset & (self.PIN_CHUNK_SIZE - 1) != 0:
            raise ValueError("Offset must be aligned to PIN_CHUNK_SIZE")
        if size & (self.PIN_CHUNK_SIZE - 1) != 0:
            raise ValueError("Size must be aligned to PIN_CHUNK_SIZE")
        if offset + size > self._final_size:
            raise ValueError("Pinning exceeds buffer size")

        ptr = self._buffer.data_ptr() + offset
        # Pin in the bound device's context so the CUDA context lands on the
        # worker GPU. Use flag: cudaHostRegisterMapped (0x02)
        with torch_dev.device(self._pin_device):
            pinned = torch_dev.ext.pin_memory(ptr, size, 2)
        if not pinned:
            raise RuntimeError(
                f"pin_memory failed for chunk at ptr={ptr:#x} size={size}"
            )
        self._pin_record.append((ptr, size))

    def _commit_expansion(self, expand_size: int):
        """
        Call sbrk in the address manager to commit the expansion.
        """
        self._address_manager.sbrk(expand_size)

    def _log_expansion_progress(self, expanded_since_last_log: int):
        """
        Log the cumulative expansion progress since the last log.
        """
        percent = 100.0 * self._curr_size / self._final_size
        logger.info(
            "LazyMemoryAllocator: Expanded %s MB pinned memory, "
            "now total is %s MB / %s MB (%.1f%%)",
            expanded_since_last_log >> 20,
            self._curr_size >> 20,
            self._final_size >> 20,
            percent,
        )

    def _expand_loop(self):
        """
        Expand the pinned pool from the init chunk up to the final size.

        Commits (sbrk) only the successfully pinned bytes of each batch, so
        the usable address space never runs ahead of the pinned bytes.
        """
        last_log_size = self._curr_size
        while self._curr_size < self._final_size and not self._stop_pinning.is_set():
            # Expand chunk by chunk and commit
            for _ in range(self.COMMIT_SIZE // self.PIN_CHUNK_SIZE):
                if self._curr_size >= self._final_size or self._stop_pinning.is_set():
                    break
                self._pin_memory_chunk(self._curr_size, self.PIN_CHUNK_SIZE)
                self._curr_size += self.PIN_CHUNK_SIZE

            expand_size = self._curr_size - self._committed_size
            if expand_size > 0:
                self._commit_expansion(expand_size)
                self._committed_size = self._curr_size

            # Log every LOG_INTERVAL bytes, and always on the final commit.
            expanded_since_last_log = self._curr_size - last_log_size
            if (
                expanded_since_last_log >= self.LOG_INTERVAL
                or self._curr_size >= self._final_size
            ):
                self._log_expansion_progress(expanded_since_last_log)
                last_log_size = self._curr_size
