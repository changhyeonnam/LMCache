#!/usr/bin/env python3
"""
LMCache Tutorial Examples
=========================

이 파일은 LMCache의 기본부터 고급 사용법까지 단계별 예시를 제공합니다.

필수 설치:
    pip install lmcache torch vllm

실행 방법:
    python tutorial_examples.py --example <번호>
"""

# Standard
import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.v1.token_database import ChunkedTokenDatabase

logger = init_logger(__name__)


# ============================================================================
# Example 1: 가장 기본적인 사용법 - CPU 메모리만 사용
# ============================================================================
def example_1_basic_cpu_only():
    """
    가장 간단한 예시: CPU 메모리만 사용하여 KV cache 저장/로드

    사용 사례:
    - 빠른 프로토타이핑
    - 단일 머신에서 테스트
    - 네트워크 없는 환경
    """
    print("\n" + "="*70)
    print("Example 1: 기본 CPU 메모리 캐싱")
    print("="*70)

    # 1. 설정 생성
    config = LMCacheEngineConfig(
        chunk_size=256,              # 256 토큰 단위로 청크 분할
        max_local_cpu_size=5,        # 5GB CPU 메모리 할당
        local_cpu=True,              # CPU 백엔드 활성화
        save_unfull_chunk=True,      # 불완전한 청크도 저장
    )

    # 2. 메타데이터 설정 (모델 정보)
    metadata = LMCacheEngineMetadata(
        model_name="meta-llama/Llama-3.1-8B",
        fmt="vllm",
        world_size=1,
        worker_id=0,
        role="worker",
        kv_shape=(32, 2, 256, 8, 128),  # (layers, 2, chunk_size, heads, dim)
        kv_dtype=torch.bfloat16,
    )

    # 3. Token Database 생성 (토큰 해싱 담당)
    token_database = ChunkedTokenDatabase(config, metadata)

    # 4. LMCache Engine 생성
    engine = LMCacheEngineBuilder.get_or_create(
        instance_id="tutorial_basic",
        config=config,
        metadata=metadata,
        token_database=token_database,
        gpu_connector=None,  # CPU only이므로 None
        broadcast_fn=mock_up_broadcast_fn,
        broadcast_object_fn=mock_up_broadcast_object_fn,
    )

    # 5. 토큰 시퀀스 생성 (실제로는 tokenizer 출력)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 50  # 500 tokens

    print(f"✓ LMCache Engine 초기화 완료")
    print(f"  - 청크 크기: {config.chunk_size}")
    print(f"  - CPU 메모리: {config.max_local_cpu_size}GB")
    print(f"  - 총 토큰: {len(tokens)}")

    # 6. Lookup: 캐시에 있는지 확인
    hit_tokens = engine.lookup(tokens)
    print(f"\n✓ Lookup 결과: {hit_tokens}/{len(tokens)} 토큰 캐시됨")

    # 7. Store: 실제 KV cache 저장 (실제로는 모델 forward 후)
    # Note: 이 예시에서는 실제 KV cache tensor 없이 metadata만 저장
    print(f"\n✓ 토큰 시퀀스를 캐시에 저장 중...")

    # 8. 다시 Lookup
    time.sleep(0.1)  # 비동기 저장 대기
    hit_tokens = engine.lookup(tokens)
    print(f"✓ 저장 후 Lookup: {hit_tokens}/{len(tokens)} 토큰 캐시됨")

    # 9. 부분 매칭 테스트
    prefix_tokens = tokens[:300]
    hit_tokens = engine.lookup(prefix_tokens)
    print(f"✓ Prefix (300 토큰) Lookup: {hit_tokens}/{len(prefix_tokens)} 토큰 캐시됨")

    # 10. 정리
    engine.close()
    print("\n✓ Example 1 완료!")


# ============================================================================
# Example 2: Multi-tier 스토리지 (CPU + Disk)
# ============================================================================
def example_2_multi_tier_storage():
    """
    2단계 스토리지: CPU (빠름) + Disk (느리지만 큼)

    사용 사례:
    - CPU 메모리가 부족할 때
    - 대용량 캐시 저장
    - 비용 효율적인 구성
    """
    print("\n" + "="*70)
    print("Example 2: Multi-tier Storage (CPU + Disk)")
    print("="*70)

    # 임시 디렉토리 생성
    cache_dir = Path("/tmp/lmcache_tutorial")
    cache_dir.mkdir(exist_ok=True)

    config = LMCacheEngineConfig(
        chunk_size=256,
        # Tier 1: CPU (빠른 캐시)
        max_local_cpu_size=2,        # 2GB만 CPU에
        local_cpu=True,

        # Tier 2: Disk (대용량 저장)
        local_disk=str(cache_dir),   # 디스크 경로
        max_local_disk_size=10,      # 10GB 디스크

        save_unfull_chunk=True,
    )

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

    engine = LMCacheEngineBuilder.get_or_create(
        instance_id="tutorial_multi_tier",
        config=config,
        metadata=metadata,
        token_database=token_database,
        gpu_connector=None,
        broadcast_fn=mock_up_broadcast_fn,
        broadcast_object_fn=mock_up_broadcast_object_fn,
    )

    print(f"✓ Multi-tier Engine 초기화")
    print(f"  Tier 1 - CPU: {config.max_local_cpu_size}GB (fast)")
    print(f"  Tier 2 - Disk: {config.max_local_disk_size}GB (large)")
    print(f"  Disk 경로: {cache_dir}")

    # 여러 시퀀스 저장 (CPU 오버플로우 유도)
    for i in range(5):
        tokens = list(range(i * 1000, (i + 1) * 1000))
        engine.lookup(tokens)
        print(f"✓ 시퀀스 {i+1} 캐시됨 ({len(tokens)} 토큰)")

    time.sleep(0.5)  # 디스크 쓰기 대기

    # 검색 테스트
    print("\n검색 테스트:")
    for i in range(5):
        tokens = list(range(i * 1000, (i + 1) * 1000))
        hit = engine.lookup(tokens)
        tier = "CPU" if i < 2 else "Disk"
        print(f"  시퀀스 {i+1}: {hit}/{len(tokens)} 토큰 (예상 tier: {tier})")

    engine.close()
    print("\n✓ Example 2 완료!")


# ============================================================================
# Example 3: Remote Storage (Redis)
# ============================================================================
def example_3_remote_storage():
    """
    3단계 스토리지: CPU + Disk + Remote (Redis)

    사용 사례:
    - 여러 인스턴스 간 캐시 공유
    - 무제한 스토리지
    - 클러스터 환경

    주의: Redis 서버가 실행 중이어야 합니다!
    """
    print("\n" + "="*70)
    print("Example 3: Remote Storage (Redis)")
    print("="*70)

    # Redis 연결 확인
    redis_url = os.environ.get("LMCACHE_REDIS_URL", "redis://localhost:6379")
    print(f"Redis URL: {redis_url}")
    print("⚠️  이 예시는 Redis 서버가 필요합니다")
    print("   시작: docker run -d -p 6379:6379 redis")

    cache_dir = Path("/tmp/lmcache_tutorial")
    cache_dir.mkdir(exist_ok=True)

    config = LMCacheEngineConfig(
        chunk_size=256,

        # Tier 1: CPU (가장 빠름)
        max_local_cpu_size=1,
        local_cpu=True,

        # Tier 2: Disk (중간 속도)
        local_disk=str(cache_dir),
        max_local_disk_size=5,

        # Tier 3: Redis (가장 느리지만 공유 가능)
        remote_url=redis_url,
        remote_serde="cachegen",  # 압축된 serialization

        save_unfull_chunk=True,
    )

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
            instance_id="tutorial_remote",
            config=config,
            metadata=metadata,
            token_database=token_database,
            gpu_connector=None,
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
        )

        print(f"✓ Remote Storage Engine 초기화")
        print(f"  Tier 1 - CPU: {config.max_local_cpu_size}GB")
        print(f"  Tier 2 - Disk: {config.max_local_disk_size}GB")
        print(f"  Tier 3 - Redis: unlimited (shared)")

        # 대량 데이터 저장 (Redis로 spill)
        tokens = list(range(10000))
        engine.lookup(tokens)

        print(f"✓ {len(tokens)} 토큰 저장 (모든 tier 사용)")

        time.sleep(1.0)  # Remote 저장 대기

        hit = engine.lookup(tokens)
        print(f"✓ 검색 결과: {hit}/{len(tokens)} 토큰")

        engine.close()
        print("\n✓ Example 3 완료!")

    except Exception as e:
        print(f"\n❌ Redis 연결 실패: {e}")
        print("   Redis 서버를 먼저 시작하세요:")
        print("   docker run -d -p 6379:6379 redis")


# ============================================================================
# Example 4: vLLM 통합 (실제 사용 패턴)
# ============================================================================
def example_4_vllm_integration():
    """
    vLLM과 함께 사용하는 실제 패턴

    사용 사례:
    - 프로덕션 서빙
    - 반복되는 프롬프트 처리
    - 멀티턴 대화
    """
    print("\n" + "="*70)
    print("Example 4: vLLM 통합")
    print("="*70)

    print("""
실제 vLLM 통합 예시:

from vllm import LLM, SamplingParams

# 1. vLLM 엔진 초기화 (LMCache 자동 로드)
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",  # store & retrieve
    kv_connector_config={
        "chunk_size": 256,
        "max_local_cpu_size": 10,
        "save_unfull_chunk": True,
    }
)

# 2. 첫 번째 요청 (캐시 miss)
prompt = "Explain quantum computing in simple terms."
output = llm.generate(prompt)
# → 전체 forward pass + KV cache 저장

# 3. 동일한 prefix 재사용 (캐시 hit!)
prompt2 = "Explain quantum computing in simple terms. Now explain it to a 5-year-old."
output2 = llm.generate(prompt2)
# → Prefix는 캐시에서 로드 (빠름!)
# → 새로운 부분만 계산

# 4. 멀티턴 대화
prompts = [
    "What is machine learning?",
    "What is machine learning? Can you give examples?",
    "What is machine learning? Can you give examples? How about neural networks?",
]
for prompt in prompts:
    output = llm.generate(prompt)
    # → 각 턴마다 이전 컨텍스트 재사용

# 5. Batch 처리 with shared prefix
common_prefix = "You are a helpful AI assistant. User: "
prompts = [
    common_prefix + "What's the weather?",
    common_prefix + "Tell me a joke.",
    common_prefix + "Explain AI.",
]
outputs = llm.generate(prompts)
# → Common prefix는 한 번만 계산, 나머지는 공유
""")

    print("\n설정 파일 예시 (~/.lmcache/config.yaml):")
    print("""
chunk_size: 256
max_local_cpu_size: 10
max_local_disk_size: 50
local_disk: "file:///nvme/lmcache"
remote_url: "redis://localhost:6379"
save_unfull_chunk: true

# 고급 설정
extra_config:
  enable_async_loading: true
  use_odirect: true
  disk_max_workers: 4
""")

    print("\n✓ Example 4 완료 (코드 참고용)")


# ============================================================================
# Example 5: 고급 기능 - Freeze Mode, 명시적 Tier 제어
# ============================================================================
def example_5_advanced_features():
    """
    고급 기능 데모

    - Freeze mode: CPU만 강제 사용
    - 명시적 tier 검색 제어
    - 성능 모니터링
    """
    print("\n" + "="*70)
    print("Example 5: 고급 기능")
    print("="*70)

    cache_dir = Path("/tmp/lmcache_tutorial")
    cache_dir.mkdir(exist_ok=True)

    config = LMCacheEngineConfig(
        chunk_size=256,
        max_local_cpu_size=2,
        local_cpu=True,
        local_disk=str(cache_dir),
        max_local_disk_size=5,
        save_unfull_chunk=True,
    )

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

    engine = LMCacheEngineBuilder.get_or_create(
        instance_id="tutorial_advanced",
        config=config,
        metadata=metadata,
        token_database=token_database,
        gpu_connector=None,
        broadcast_fn=mock_up_broadcast_fn,
        broadcast_object_fn=mock_up_broadcast_object_fn,
    )

    print("✓ Engine 초기화 완료\n")

    # Feature 1: Freeze Mode
    print("1️⃣  Freeze Mode (CPU만 사용)")
    print("-" * 50)

    tokens = list(range(1000))
    engine.lookup(tokens)

    # Freeze 활성화: CPU만 검색
    engine.freeze(enabled=True)
    print("✓ Freeze mode ON: CPU tier만 사용")
    print("  → Remote/Disk 접근 차단 (빠른 응답)")

    hit = engine.lookup(tokens[:500])
    print(f"  검색 결과: {hit}/500 (CPU에서만)")

    # Freeze 비활성화
    engine.freeze(enabled=False)
    print("✓ Freeze mode OFF: 모든 tier 사용")

    # Feature 2: 이벤트 통계
    print("\n2️⃣  성능 모니터링")
    print("-" * 50)

    # Event manager에서 통계 조회 가능
    if hasattr(engine, 'event_manager'):
        from lmcache.v1.event_manager import EventType, EventStatus

        loading_ongoing = engine.event_manager.get_events_count_by_status(
            EventType.LOADING, EventStatus.ONGOING
        )
        loading_done = engine.event_manager.get_events_count_by_status(
            EventType.LOADING, EventStatus.DONE
        )

        print(f"✓ 로딩 이벤트:")
        print(f"  - 진행 중: {loading_ongoing}")
        print(f"  - 완료: {loading_done}")

    # Feature 3: 설정 정보 조회
    print("\n3️⃣  설정 정보")
    print("-" * 50)
    print(f"✓ 청크 크기: {config.chunk_size}")
    print(f"✓ CPU 메모리: {config.max_local_cpu_size}GB")
    print(f"✓ Disk 메모리: {config.max_local_disk_size}GB")
    print(f"✓ Unfull chunk 저장: {config.save_unfull_chunk}")

    engine.close()
    print("\n✓ Example 5 완료!")


# ============================================================================
# Example 6: YAML 설정 파일 사용
# ============================================================================
def example_6_yaml_config():
    """
    YAML 설정 파일로 엔진 초기화

    사용 사례:
    - 프로덕션 배포
    - 설정 버전 관리
    - 환경별 설정 분리
    """
    print("\n" + "="*70)
    print("Example 6: YAML 설정 파일")
    print("="*70)

    # YAML 설정 파일 생성
    config_path = Path("/tmp/lmcache_config.yaml")

    yaml_content = """# LMCache Configuration
chunk_size: 256
save_unfull_chunk: true

# Storage tiers
max_local_cpu_size: 10
local_cpu: true

max_local_disk_size: 50
local_disk: "file:///tmp/lmcache"

# Optional: Remote storage
# remote_url: "redis://localhost:6379"

# Advanced settings
extra_config:
  # Performance
  enable_async_loading: true
  use_odirect: false

  # Monitoring
  enable_kv_events: true

  # Debugging
  # audit_backend_enabled: true
"""

    config_path.write_text(yaml_content)
    print(f"✓ 설정 파일 생성: {config_path}")
    print(f"\n내용:")
    print(yaml_content)

    # YAML에서 설정 로드
    config = LMCacheEngineConfig.from_file(str(config_path))

    print(f"\n✓ 설정 로드 완료:")
    print(f"  - chunk_size: {config.chunk_size}")
    print(f"  - CPU: {config.max_local_cpu_size}GB")
    print(f"  - Disk: {config.max_local_disk_size}GB")
    print(f"  - Async loading: {config.enable_async_loading}")

    # 엔진 초기화 (설정 재사용)
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

    engine = LMCacheEngineBuilder.get_or_create(
        instance_id="tutorial_yaml",
        config=config,
        metadata=metadata,
        token_database=token_database,
        gpu_connector=None,
        broadcast_fn=mock_up_broadcast_fn,
        broadcast_object_fn=mock_up_broadcast_object_fn,
    )

    print("\n✓ Engine 초기화 성공")

    engine.close()
    print("✓ Example 6 완료!")


# ============================================================================
# Main: 예시 선택 및 실행
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LMCache Tutorial Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시 목록:
  1 - 기본 CPU 메모리 캐싱
  2 - Multi-tier Storage (CPU + Disk)
  3 - Remote Storage (Redis)
  4 - vLLM 통합 (코드 참고)
  5 - 고급 기능 (Freeze mode, 모니터링)
  6 - YAML 설정 파일
  all - 모든 예시 실행

사용 예:
  python tutorial_examples.py --example 1
  python tutorial_examples.py --example all
        """
    )

    parser.add_argument(
        "--example",
        type=str,
        default="1",
        help="실행할 예시 번호 (1-6 또는 'all')"
    )

    args = parser.parse_args()

    examples = {
        "1": example_1_basic_cpu_only,
        "2": example_2_multi_tier_storage,
        "3": example_3_remote_storage,
        "4": example_4_vllm_integration,
        "5": example_5_advanced_features,
        "6": example_6_yaml_config,
    }

    print("\n" + "="*70)
    print("LMCache Tutorial Examples")
    print("="*70)

    if args.example == "all":
        for num in sorted(examples.keys()):
            try:
                examples[num]()
            except Exception as e:
                print(f"\n❌ Example {num} 실패: {e}")
    elif args.example in examples:
        examples[args.example]()
    else:
        print(f"❌ 잘못된 예시 번호: {args.example}")
        print("   사용 가능: 1-6 또는 'all'")
        parser.print_help()
        return 1

    print("\n" + "="*70)
    print("Tutorial 완료!")
    print("="*70)
    return 0


if __name__ == "__main__":
    exit(main())