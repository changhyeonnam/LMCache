# LMCache 튜토리얼 가이드

이 가이드는 LMCache를 처음 사용하는 분들을 위한 단계별 튜토리얼입니다.

## 📚 목차

1. [빠른 시작](#빠른-시작)
2. [예시 목록](#예시-목록)
3. [실전 사용 패턴](#실전-사용-패턴)
4. [트러블슈팅](#트러블슈팅)

---

## 🚀 빠른 시작

### 설치

```bash
# LMCache 및 의존성 설치
pip install lmcache torch vllm

# 또는 소스에서 설치
cd /path/to/LMCache
pip install -e .
```

### 첫 실행

```bash
# 가장 기본적인 예시 실행
cd examples
python tutorial_examples.py --example 1
```

---

## 📖 예시 목록

### Example 1: 기본 CPU 메모리 캐싱 ⭐ 초보자 추천

가장 간단한 시작점. CPU 메모리만 사용하여 KV cache를 저장하고 불러옵니다.

```bash
python tutorial_examples.py --example 1
```

**배우는 내용:**
- LMCache Engine 초기화
- 설정(Config) 생성
- Lookup/Store 기본 API
- Token 시퀀스 캐싱

**사용 사례:**
- 빠른 프로토타이핑
- 로컬 개발 환경
- 단순한 캐싱 요구사항

---

### Example 2: Multi-tier Storage (CPU + Disk)

2단계 스토리지를 사용하여 CPU 메모리를 넘어서는 대용량 캐시를 관리합니다.

```bash
python tutorial_examples.py --example 2
```

**배우는 내용:**
- 여러 tier 구성
- 자동 tier fallback
- Disk 스토리지 설정

**사용 사례:**
- CPU 메모리가 부족할 때
- 대용량 모델 서빙
- 비용 효율적인 구성

---

### Example 3: Remote Storage (Redis)

Redis를 사용하여 여러 인스턴스 간 캐시를 공유합니다.

```bash
# Redis 서버 시작 필요
docker run -d -p 6379:6379 redis

python tutorial_examples.py --example 3
```

**배우는 내용:**
- Remote backend 설정
- 3-tier 아키텍처 (CPU → Disk → Redis)
- 클러스터 환경 캐시 공유

**사용 사례:**
- 분산 서빙 환경
- 여러 GPU 노드 간 캐시 공유
- 무제한 스토리지 필요 시

---

### Example 4: vLLM 통합 (코드 참고)

실제 vLLM과 함께 사용하는 프로덕션 패턴을 보여줍니다.

```bash
python tutorial_examples.py --example 4
```

**배우는 내용:**
- vLLM connector 설정
- 멀티턴 대화 캐싱
- Batch 요청 with shared prefix
- 프로덕션 설정 예시

**사용 사례:**
- LLM 서빙 서버
- 챗봇 애플리케이션
- RAG 시스템

---

### Example 5: 고급 기능

Freeze mode, 성능 모니터링 등 고급 기능을 데모합니다.

```bash
python tutorial_examples.py --example 5
```

**배우는 내용:**
- Freeze mode (CPU only 강제)
- 이벤트 모니터링
- 설정 정보 조회
- 성능 최적화 팁

**사용 사례:**
- 네트워크 장애 대응
- 성능 튜닝
- 프로덕션 운영

---

### Example 6: YAML 설정 파일

YAML 파일로 설정을 관리하는 방법을 보여줍니다.

```bash
python tutorial_examples.py --example 6
```

**배우는 내용:**
- YAML 설정 파일 작성
- 파일에서 설정 로드
- 환경별 설정 관리

**사용 사례:**
- 프로덕션 배포
- 설정 버전 관리
- DevOps 통합

---

### 모든 예시 실행

```bash
python tutorial_examples.py --example all
```

---

## 💡 실전 사용 패턴

### 패턴 1: 로컬 개발 환경

```yaml
# config.yaml
chunk_size: 256
max_local_cpu_size: 5
local_cpu: true
save_unfull_chunk: true
```

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
)

# 간단한 테스트
output = llm.generate("Hello, world!")
```

---

### 패턴 2: 프로덕션 서버 (Single Node)

```yaml
# production.yaml
chunk_size: 256

# 3-tier 구성
max_local_cpu_size: 20      # Hot cache
local_cpu: true

max_local_disk_size: 100    # Warm cache
local_disk: "file:///nvme/lmcache"

remote_url: "redis://redis-cluster:6379"  # Cold cache

# 성능 최적화
extra_config:
  enable_async_loading: true
  use_odirect: true
  disk_max_workers: 8
```

---

### 패턴 3: 분산 클러스터 (Multi-node)

```yaml
# cluster.yaml
chunk_size: 256
max_local_cpu_size: 10
local_cpu: true

# P2P 캐시 공유
enable_p2p: true

# 중앙 Redis
remote_url: "redis://redis-cluster:6379"

# Controller 활성화
enable_controller: true
```

**실행:**
```bash
# Node 1
export LMCACHE_INSTANCE_ID=node1
vllm serve meta-llama/Llama-3.1-8B \
  --kv-connector LMCacheConnectorV1 \
  --kv-role kv_both

# Node 2
export LMCACHE_INSTANCE_ID=node2
vllm serve meta-llama/Llama-3.1-8B \
  --kv-connector LMCacheConnectorV1 \
  --kv-role kv_both
```

---

### 패턴 4: NFS를 추가 Tier로 사용

```bash
# NFS 마운트
sudo mount -t nfs \
  -o rsize=1048576,wsize=1048576 \
  nfs-server:/export/cache /mnt/nfs-cache
```

```yaml
# nfs_config.yaml
chunk_size: 256

# Tier 1: CPU (fastest)
max_local_cpu_size: 10
local_cpu: true

# Tier 2: Local NVMe
max_local_disk_size: 50
local_disk: "file:///nvme/lmcache"

# Tier 3: NFS (large, shared)
remote_url: "fs://host:0/mnt/nfs-cache"

extra_config:
  fs_connector_read_ahead_size: 8192
```

---

### 패턴 5: Freeze Mode를 활용한 장애 대응

```python
from lmcache.v1.cache_engine import LMCacheEngineBuilder

engine = LMCacheEngineBuilder.get_instance("my_instance")

try:
    # 정상 작동: 모든 tier 사용
    result = engine.lookup(tokens)
except NetworkError:
    # 네트워크 장애 발생!
    # → CPU만 사용하도록 전환
    engine.freeze(enabled=True)
    print("⚠️  Freeze mode 활성화: CPU tier만 사용")

    result = engine.lookup(tokens)  # CPU에서만 검색

# 네트워크 복구 시
engine.freeze(enabled=False)
print("✓ Freeze mode 해제: 모든 tier 복구")
```

---

## 🛠️ 트러블슈팅

### 문제 1: "Redis connection failed"

**원인:** Redis 서버가 실행되지 않음

**해결:**
```bash
# Redis 시작
docker run -d -p 6379:6379 redis

# 또는 로컬 설치
redis-server
```

---

### 문제 2: "CUDA out of memory"

**원인:** GPU 메모리 부족

**해결:**
```yaml
# GPU 버퍼 줄이기
extra_config:
  use_gpu_buffer: false  # GPU 중간 버퍼 비활성화
```

또는 CPU only 모드 사용:
```python
gpu_connector=None  # CPU only
```

---

### 문제 3: "Permission denied" (Disk)

**원인:** 디스크 경로 권한 문제

**해결:**
```bash
# 캐시 디렉토리 권한 수정
sudo mkdir -p /nvme/lmcache
sudo chown $USER:$USER /nvme/lmcache
chmod 755 /nvme/lmcache
```

---

### 문제 4: "Cache hit rate too low"

**원인:** 청크 크기 또는 설정 문제

**해결:**
```yaml
# chunk_size 조정
chunk_size: 256  # 또는 512

# Unfull chunk 저장 활성화
save_unfull_chunk: true

# 더 큰 CPU 메모리
max_local_cpu_size: 20
```

---

### 문제 5: "Slow performance"

**체크리스트:**

1. **Async loading 활성화**
   ```yaml
   extra_config:
     enable_async_loading: true
   ```

2. **O_DIRECT 사용** (Linux)
   ```yaml
   extra_config:
     use_odirect: true
   ```

3. **Disk worker 수 증가**
   ```yaml
   extra_config:
     disk_max_workers: 8
   ```

4. **Prefetching 활성화**
   ```python
   # vLLM에서 prefetch 사용
   llm.generate(prompt, prefetch=True)
   ```

---

## 📊 성능 벤치마크

### 예상 성능 (참고용)

| Tier | Latency | Throughput | 용량 |
|------|---------|------------|------|
| CPU | 0.1-1ms | 10GB/s | 10-100GB |
| Local NVMe | 1-5ms | 3GB/s | 100GB-1TB |
| NFS | 5-20ms | 100MB/s | Unlimited |
| Redis | 10-50ms | 50MB/s | Unlimited |

### 캐시 히트율 vs 성능

- **90%+ hit rate**: ~10x speedup
- **70-90% hit rate**: ~5x speedup
- **50-70% hit rate**: ~2x speedup
- **<50% hit rate**: 설정 최적화 필요

---

## 📚 더 알아보기

### 공식 문서
- [LMCache GitHub](https://github.com/LMCache/LMCache)
- [vLLM Documentation](https://docs.vllm.ai/)

### 고급 주제
- P2P 캐시 공유: `examples/kv_cache_reuse/share_across_instances/p2p_sharing/`
- 캐시 컨트롤러: `examples/cache_controller/`
- 커스텀 플러그인: `examples/runtime_plugins/`

### 커뮤니티
- GitHub Issues: 버그 리포트 및 기능 요청
- Discussions: 질문 및 아이디어 공유

---

## 🎯 다음 단계

1. ✅ **기본 예시 실행** (`example 1`)
2. ✅ **Multi-tier 설정** (`example 2`)
3. ✅ **vLLM 통합** (`example 4`)
4. 📝 **프로덕션 설정 작성**
5. 🚀 **성능 최적화 및 모니터링**

Happy caching! 🎉