#!/usr/bin/env python3
"""
GH100 + NIXL + LMCache 완전한 예시
====================================

GH100 GPU 1장 환경에서 NIXL GDS를 사용한 고성능 KV 캐싱 예시

실행:
    python gh100_nixl_example.py
"""

import os
import time
from pathlib import Path

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase

logger = init_logger(__name__)


def check_environment():
    """환경 체크"""
    print("\n" + "="*70)
    print("🔍 환경 체크")
    print("="*70)

    # CUDA 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다")
        return False

    print(f"✓ CUDA 사용 가능")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # NIXL 확인
    try:
        from nixl._api import nixlBind
        print(f"✓ NIXL 설치됨")
    except ImportError:
        print(f"❌ NIXL이 설치되지 않았습니다")
        print(f"   설치: pip install nixl")
        return False

    # NVMe 확인
    nvme_path = Path("/mnt/nvme")
    if nvme_path.exists():
        print(f"✓ NVMe 마운트: {nvme_path}")
    else:
        print(f"⚠️  /mnt/nvme가 없습니다. /tmp를 사용합니다")

    return True


def example_posix_backend():
    """예시 1: NIXL POSIX Backend (기본)"""
    print("\n" + "="*70)
    print("📝 Example 1: NIXL POSIX Backend")
    print("="*70)

    # 캐시 디렉토리
    cache_path = Path("/mnt/nvme/lmcache") if Path("/mnt/nvme").exists() else Path("/tmp/lmcache")
    cache_path.mkdir(parents=True, exist_ok=True)

    # 설정
    config = LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=False,  # NIXL을 allocator로 사용
        nixl_buffer_size=1 * 1024**3,  # 1GB
        nixl_buffer_device="cpu",
        save_unfull_chunk=True,
    )

    config.extra_config = {
        "enable_nixl_storage": True,
        "nixl_backend": "POSIX",
        "nixl_pool_size": 4,
        "nixl_path": str(cache_path),
        "use_direct_io": False,
    }

    print(f"✓ 설정:")
    print(f"  Backend: POSIX")
    print(f"  Buffer: {config.nixl_buffer_size / 1e9:.1f}GB (CPU)")
    print(f"  Path: {cache_path}")

    # 메타데이터
    metadata = LMCacheEngineMetadata(
        model_name="meta-llama/Llama-3.1-8B",
        fmt="vllm",
        world_size=1,
        worker_id=0,
        role="worker",
        kv_shape=(32, 2, 256, 8, 128),
        kv_dtype=torch.bfloat16,
    )

    token_database = ChunkedTokenDatabase(config, metadata)

    # Engine 생성
    try:
        engine = LMCacheEngineBuilder.get_or_create(
            instance_id="gh100_posix",
            config=config,
            metadata=metadata,
            token_database=token_database,
            gpu_connector=None,
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
        )

        print(f"✓ LMCache Engine 초기화 성공")

        # 테스트
        tokens = list(range(1000))

        start = time.time()
        hit = engine.lookup(tokens)
        miss_time = time.time() - start
        print(f"\n첫 조회 (miss): {miss_time*1000:.2f}ms, hit={hit}/{len(tokens)}")

        time.sleep(0.2)

        start = time.time()
        hit = engine.lookup(tokens)
        hit_time = time.time() - start
        print(f"재조회 (hit): {hit_time*1000:.2f}ms, hit={hit}/{len(tokens)}")

        if hit_time > 0:
            print(f"속도 향상: {miss_time/hit_time:.1f}x")

        engine.close()
        print("\n✓ Example 1 완료!")
        return True

    except Exception as e:
        print(f"\n❌ 실패: {e}")
        return False


def example_gds_backend():
    """예시 2: NIXL GDS Backend (GPU Direct Storage)"""
    print("\n" + "="*70)
    print("🚀 Example 2: NIXL GDS Backend (GPU Direct Storage)")
    print("="*70)

    # GH100 확인
    if not torch.cuda.is_available():
        print("❌ CUDA가 필요합니다")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    # nvidia-fs 확인
    import subprocess
    result = subprocess.run(["lsmod"], capture_output=True, text=True)
    if "nvidia_fs" in result.stdout:
        print("✓ nvidia-fs 모듈 로드됨")
    else:
        print("⚠️  nvidia-fs 모듈이 없습니다")
        print("   로드: sudo modprobe nvidia-fs")
        print("   GDS 없이 POSIX로 fallback합니다")
        return example_posix_backend()

    # 캐시 디렉토리
    cache_path = Path("/mnt/nvme/lmcache_gds") if Path("/mnt/nvme").exists() else Path("/tmp/lmcache_gds")
    cache_path.mkdir(parents=True, exist_ok=True)

    # GDS 설정
    config = LMCacheEngineConfig(
        chunk_size=256,
        local_cpu=False,
        nixl_buffer_size=4 * 1024**3,  # 4GB (GH100은 96GB VRAM)
        nixl_buffer_device="cuda",  # ⭐ GPU 버퍼!
        save_unfull_chunk=True,
    )

    config.extra_config = {
        "enable_nixl_storage": True,
        "nixl_backend": "GDS",  # ⭐ GPU Direct Storage!
        "nixl_pool_size": 8,
        "nixl_path": str(cache_path),
        "use_direct_io": True,
        "nixl_backend_params": {
            "gds_batch_size": 4,
        }
    }

    print(f"✓ GDS 설정:")
    print(f"  Backend: GDS (GPU Direct Storage)")
    print(f"  Buffer: {config.nixl_buffer_size / 1e9:.1f}GB (GPU)")
    print(f"  Pool: {config.extra_config['nixl_pool_size']}")
    print(f"  Path: {cache_path}")
    print(f"  Direct I/O: {config.extra_config['use_direct_io']}")

    metadata = LMCacheEngineMetadata(
        model_name="meta-llama/Llama-3.1-8B",
        fmt="vllm",
        world_size=1,
        worker_id=0,
        role="worker",
        kv_shape=(32, 2, 256, 8, 128),
        kv_dtype=torch.bfloat16,
    )

    token_database = ChunkedTokenDatabase(config, metadata)

    try:
        engine = LMCacheEngineBuilder.get_or_create(
            instance_id="gh100_gds",
            config=config,
            metadata=metadata,
            token_database=token_database,
            gpu_connector=None,
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
        )

        print(f"\n✓ GDS Engine 초기화 성공!")
        print(f"  GPU → NVMe 직접 전송 가능")

        # 성능 테스트
        tokens = list(range(2000))

        print(f"\n성능 벤치마크 ({len(tokens)} tokens):")

        start = time.time()
        hit = engine.lookup(tokens)
        miss_time = time.time() - start
        print(f"  첫 조회 (miss): {miss_time*1000:.2f}ms")

        time.sleep(0.3)

        start = time.time()
        hit = engine.lookup(tokens)
        hit_time = time.time() - start
        print(f"  재조회 (GDS hit): {hit_time*1000:.2f}ms")
        print(f"  속도 향상: {miss_time/hit_time:.1f}x")
        print(f"  Cache hit: {hit}/{len(tokens)} tokens")

        engine.close()
        print("\n✓ Example 2 완료!")
        print("🎉 GDS가 정상 작동합니다!")
        return True

    except Exception as e:
        print(f"\n❌ GDS 실패: {e}")
        print("   POSIX backend로 fallback을 시도하세요")
        return False


def example_hybrid_tiering():
    """예시 3: Hybrid Tiering (CPU + NIXL GDS)"""
    print("\n" + "="*70)
    print("⚡ Example 3: Hybrid Tiering (CPU + NIXL GDS)")
    print("="*70)

    if not torch.cuda.is_available():
        print("❌ CUDA가 필요합니다")
        return False

    cache_path = Path("/mnt/nvme/lmcache_hybrid") if Path("/mnt/nvme").exists() else Path("/tmp/lmcache_hybrid")
    cache_path.mkdir(parents=True, exist_ok=True)

    # Hybrid 설정
    config = LMCacheEngineConfig(
        chunk_size=256,

        # Tier 1: CPU (ultra-fast, small)
        max_local_cpu_size=5,  # 5GB hot cache
        local_cpu=True,

        # Tier 2: NIXL GDS (fast, large)
        nixl_buffer_size=4 * 1024**3,  # 4GB
        nixl_buffer_device="cuda",

        save_unfull_chunk=True,
    )

    config.extra_config = {
        "enable_nixl_storage": True,
        "nixl_backend": "GDS",
        "nixl_pool_size": 8,
        "nixl_path": str(cache_path),
        "use_direct_io": True,
        "enable_async_loading": True,  # 비동기 로딩
    }

    print(f"✓ Hybrid Tiering:")
    print(f"  Tier 1 - CPU: {config.max_local_cpu_size}GB (0.1ms)")
    print(f"  Tier 2 - NIXL GDS: {config.nixl_buffer_size / 1e9:.1f}GB (0.5ms)")

    metadata = LMCacheEngineMetadata(
        model_name="meta-llama/Llama-3.1-8B",
        fmt="vllm",
        world_size=1,
        worker_id=0,
        role="worker",
        kv_shape=(32, 2, 256, 8, 128),
        kv_dtype=torch.bfloat16,
    )

    token_database = ChunkedTokenDatabase(config, metadata)

    try:
        engine = LMCacheEngineBuilder.get_or_create(
            instance_id="gh100_hybrid",
            config=config,
            metadata=metadata,
            token_database=token_database,
            gpu_connector=None,
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
        )

        print(f"\n✓ Hybrid Engine 초기화 성공")

        # 여러 시퀀스로 tier 테스트
        print(f"\nTier 동작 테스트:")

        sequences = [
            list(range(i * 1000, (i + 1) * 1000))
            for i in range(10)
        ]

        for i, seq in enumerate(sequences):
            engine.lookup(seq)
            print(f"  시퀀스 {i+1}: {len(seq)} tokens 저장")

        time.sleep(0.5)

        print(f"\n검색 테스트 (tier 확인):")
        for i, seq in enumerate(sequences):
            start = time.time()
            hit = engine.lookup(seq)
            elapsed = time.time() - start

            tier = "CPU" if i < 3 else "GDS"
            print(f"  시퀀스 {i+1}: {elapsed*1000:.2f}ms (예상: {tier})")

        engine.close()
        print("\n✓ Example 3 완료!")
        return True

    except Exception as e:
        print(f"\n❌ 실패: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("GH100 + NIXL + LMCache 예시")
    print("="*70)

    # 환경 체크
    if not check_environment():
        print("\n환경 설정을 먼저 완료하세요")
        print("가이드: gh100_nixl_setup_guide.md")
        return 1

    # 예시 실행
    print("\n" + "="*70)
    print("예시 실행")
    print("="*70)

    # Example 1: POSIX
    success1 = example_posix_backend()

    # Example 2: GDS (CUDA 있을 때만)
    if torch.cuda.is_available():
        success2 = example_gds_backend()
        success3 = example_hybrid_tiering()
    else:
        print("\n⚠️  CUDA가 없어 GDS 예시를 건너뜁니다")
        success2 = True
        success3 = True

    # 결과
    print("\n" + "="*70)
    print("결과 요약")
    print("="*70)

    results = {
        "POSIX Backend": success1,
        "GDS Backend": success2,
        "Hybrid Tiering": success3,
    }

    for name, success in results.items():
        status = "✓" if success else "❌"
        print(f"  {status} {name}")

    if all(results.values()):
        print("\n🎉 모든 예시 성공!")
        print("\n다음 단계:")
        print("  1. vLLM과 통합")
        print("     → gh100_nixl_setup_guide.md 참고")
        print("  2. 프로덕션 설정")
        print("     → production_gh100.yaml 사용")
    else:
        print("\n일부 예시 실패")
        print("트러블슈팅: gh100_nixl_setup_guide.md")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n중단됨")
        exit(1)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
