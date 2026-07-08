# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeviceIPCWrapper.device_index.

``device_index`` resolves a wrapper's ``device_uuid`` against the devices
visible to the calling process. The MP server uses it at registration time
to learn the worker's GPU (to bind the L1 pinned pool's CUDA context), so
both the success path and the not-visible failure path matter. The device
layer is mocked, so these tests need no GPU.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper


@pytest.fixture(autouse=True)
def reset_discovery_cache():
    """Isolate the class-level UUID-to-ordinal cache between tests."""
    with DeviceIPCWrapper._device_mapping_lock:
        saved = dict(DeviceIPCWrapper._discovered_device_mapping)
        DeviceIPCWrapper._discovered_device_mapping.clear()
    yield
    with DeviceIPCWrapper._device_mapping_lock:
        DeviceIPCWrapper._discovered_device_mapping.clear()
        DeviceIPCWrapper._discovered_device_mapping.update(saved)


def _wrapper_with_uuid(uuid: str) -> DeviceIPCWrapper:
    """Build a bare wrapper carrying only the interface field under test."""
    wrapper = DeviceIPCWrapper()
    wrapper.device_uuid = uuid
    return wrapper


def test_device_index_resolves_visible_uuid():
    """device_index maps the wrapper's UUID to this process's ordinal."""
    with patch("lmcache.v1.platform.base_ipc_wrapper.torch_dev") as td:
        td.is_available.return_value = True
        td.device_count.return_value = 3
        td.get_device_properties.side_effect = lambda i: SimpleNamespace(
            uuid=f"GPU-{i}"
        )

        assert _wrapper_with_uuid("GPU-2").device_index() == 2


def test_device_index_raises_for_invisible_device():
    """device_index raises when the device is not visible to this process
    (e.g. a CUDA_VISIBLE_DEVICES mismatch between worker and server)."""
    with patch("lmcache.v1.platform.base_ipc_wrapper.torch_dev") as td:
        td.is_available.return_value = True
        td.device_count.return_value = 1
        td.get_device_properties.side_effect = lambda i: SimpleNamespace(
            uuid=f"GPU-{i}"
        )

        with pytest.raises(RuntimeError, match="not found"):
            _wrapper_with_uuid("GPU-42").device_index()
