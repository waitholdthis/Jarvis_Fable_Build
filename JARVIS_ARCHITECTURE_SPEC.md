# JARVIS-Class Local AI Assistant — Technical Specification & Architecture Blueprint

**Document class:** Engineering Specification (v1.0)
**Scope:** Single-node, local-first, multimodal AI assistant with agentic OS control
**Design targets:** < 800 ms voice-to-voice latency · 100% offline primary path · concurrent LLM + VLM + ASR + TTS · sandboxed agentic execution

---

## 1. System Overview & Design Philosophy

This blueprint specifies a single high-density workstation ("the Node") that runs the entire assistant stack locally: a large reasoning LLM, a vision-language model, streaming speech recognition, streaming speech synthesis, a hybrid retrieval/memory subsystem, and a sandboxed agentic execution layer that can observe and operate the desktop.

Five non-negotiable design principles govern every decision below:

1. **Latency is the product.** Every subsystem is budgeted in milliseconds (see the end-to-end latency budget in §3.5). Anything that adds a synchronous network round-trip, a cold model load, or a blocking disk read on the hot path is rejected. All pipelines are streaming: audio is chunked, ASR emits partials, the LLM streams tokens, and TTS begins synthesis on the first clause — no stage waits for the previous stage to finish.

2. **VRAM residency over model size.** A 70B-class model that is always resident and answers in 200 ms time-to-first-token beats a 405B-class model that has to be paged in. All models are pinned in VRAM 24/7 with static allocation; there is no model swapping on the hot path.

3. **Isolation by construction.** The reasoning plane (models) never touches the execution plane (tools, shell, filesystem) directly. Every side effect flows through a policy-checked broker into a sandbox. The LLM proposes; the broker disposes.

4. **Local-first, not local-only.** The primary path is fully air-gap capable. An optional, explicitly gated escalation path can route *non-sensitive, redacted* tasks to external compute, but the system must remain fully functional with the network cable unplugged.

5. **Memory is an engineered subsystem, not a prompt hack.** Short-term, episodic, and semantic memory are separate stores with separate consistency models, write paths, and eviction policies, consolidated by a background process — not one giant vector dump.

**High-level topology:**

```
┌─────────────────────────── THE NODE ────────────────────────────┐
│                                                                 │
│  GPU 0 (96 GB) ── Primary LLM (reasoning/agent)                 │
│  GPU 1 (96 GB) ── VLM + ASR + TTS + Embedder + Reranker         │
│                                                                 │
│  ┌── Perception ──┐  ┌── Cognition ──┐  ┌── Action ──────────┐  │
│  │ mic / screen / │→ │ Orchestrator  │→ │ Tool broker →      │  │
│  │ window / input │  │ (LangGraph)   │  │ sandboxed runners  │  │
│  └────────────────┘  └───────┬───────┘  └────────────────────┘  │
│                              │                                  │
│  ┌── Memory plane ───────────┴────────────────────────────────┐ │
│  │ KV/prefix cache · Redis (ephemeral) · SQLite (episodic)    │ │
│  │ Qdrant (semantic) · Tantivy (lexical) · MinIO (blobs)      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Transport: gRPC (unix sockets) + WebSocket (UI) + ZeroMQ (hot) │
│  Egress: default-DENY nftables · audit log · optional failover  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Specification & Compute Matrix

### 2.1 Compute Node Bill of Materials

| Component | Specification | Functional Allocation |
|---|---|---|
| **GPU 0 (Reasoning)** | NVIDIA RTX PRO 6000 Blackwell Workstation Edition — 96 GB GDDR7 ECC, ~1.8 TB/s bandwidth, 600 W, PCIe 5.0 x16 | Primary LLM, always resident: weights (~65–75 GB quantized) + paged KV cache (~20 GB) for 128K context |
| **GPU 1 (Perception)** | NVIDIA RTX PRO 6000 Blackwell (second unit) — 96 GB GDDR7 ECC | VLM (~35 GB) + ASR (~3 GB) + TTS (~2–6 GB) + embedder/reranker (~8 GB) + screen-OCR + headroom for draft/router model |
| **CPU** | AMD Ryzen Threadripper PRO 9985WX — 64 cores / 128 threads, Zen 5, 128 PCIe 5.0 lanes, 8-channel memory controller | Orchestrator, vector DB, tokenization, audio DSP, sandbox VMs, RAID parity, background memory consolidation |
| **Motherboard** | WRX90 (e.g., ASUS Pro WS WRX90E-SAGE SE) — 7× PCIe 5.0 x16, 8 DIMM slots, dual 10 GbE, IPMI | Guarantees both GPUs get full x16 Gen5 (128 GB/s bidirectional each) with lanes left for NVMe |
| **System RAM** | 512 GB DDR5-6400 ECC RDIMM (8× 64 GB, all channels populated) → ~409 GB/s theoretical bandwidth | OS page cache for indices, Qdrant memory-mapped segments, CPU-offload spillover for llama.cpp fallback, sandbox VM memory |
| **Storage — Tier 0 (Models)** | 2× 4 TB PCIe 5.0 NVMe (Samsung 9100 PRO / Crucial T705, ~14 GB/s read each) in **mdadm RAID 0** → ~28 GB/s aggregate | Model weights + quantized variants. RAID 0 is acceptable: contents are reproducible artifacts. Cold-loads a 70 GB model in < 4 s |
| **Storage — Tier 1 (Data)** | 2× 4 TB PCIe 4.0 NVMe (WD SN850X / Samsung 990 PRO) in **mdadm RAID 1** | Vector DB, SQLite episodic store, audit logs, user documents. Mirrored — this data is *not* reproducible |
| **Storage — Tier 2 (Cold)** | 2× 20 TB HDD (Seagate Exos X20) in RAID 1 + ZFS with weekly scrub | Snapshots, raw session recordings, model archive, restic backup target |
| **PSU** | Seasonic PRIME TX-2200 (2200 W, Titanium) on a dedicated 20 A circuit; UPS: 3000 VA line-interactive (Eaton 9PX) | Peak draw ≈ 600 + 600 (GPUs) + 350 (CPU) + 150 (platform) ≈ 1700 W; 2200 W gives transient headroom for GPU power excursions |
| **Cooling** | CPU: 420 mm AIO or Noctua NH-U14S TR5-SP6. GPUs: blower-style workstation coolers (stock), 4× 140 mm intake / 3× 140 mm exhaust, positive pressure, dust-filtered | Sustains dual-600 W GPU load without thermal throttling; target GPU hotspot < 85 °C under 24/7 load |
| **Chassis** | Fractal Meshify 2 XL or rack-mount 4U (Sliger CX4712) | Airflow-first; rackmount preferred if colocated with UPS |
| **Audio I/O** | RØDE NT-USB+ or Shure MV7 (hardware DSP, low self-noise) + dedicated USB audio interface; optional far-field: ReSpeaker 4-Mic array | Clean 48 kHz capture; hardware AGC off (software VAD handles gating) |
| **Networking** | Dual on-board 10 GbE (disabled by default via nftables); management via IPMI on isolated VLAN | Air-gap-capable; NICs enabled only for explicit failover/update windows |

**Budget-tier substitution (same architecture, ~⅓ cost):** 2× RTX 5090 (32 GB GDDR7 each, 64 GB total) + Threadripper 7970X + 256 GB DDR5. Drops you from a 96 GB-resident 70B–120B primary model to a 32B-class primary (Qwen3-32B AWQ) with the perception stack on GPU 1. Every software decision below is unchanged.

### 2.2 GPU Allocation Strategy (No-Contention Design)

The single most common failure mode in multi-model local rigs is **time-slicing contention**: ASR, TTS, and the LLM fighting for the same SMs, producing stutter exactly when the user is speaking. The fix is hard partitioning:

| Resource | GPU 0 (96 GB) | GPU 1 (96 GB) |
|---|---|---|
| Primary LLM (e.g., GLM-4.5-Air / Llama-3.3-70B / gpt-oss-120b, quantized) | 65–75 GB weights + KV | — |
| Draft model for speculative decoding (Qwen3-4B FP8) | 4 GB | — |
| VLM (Qwen3-VL-30B-A3B or Qwen2.5-VL-32B AWQ) | — | ~24–35 GB |
| ASR (faster-whisper large-v3-turbo, INT8) | — | ~2.5 GB |
| TTS (Kokoro-82M FP16 + XTTS v2 for cloned voice) | — | ~0.5 + 4 GB |
| Embedder (Qwen3-Embedding-4B) + Reranker (BGE-reranker-v2-m3) | — | ~10 GB |
| Screen OCR / layout (PaddleOCR-v4 or Florence-2-large) | — | ~2 GB |
| Router/utility model (Qwen3-4B-Instruct, FP8) | — | 4 GB |
| Paged KV cache / headroom | ~20 GB | ~20 GB |

Enforcement mechanisms:

- **MPS (Multi-Process Service)** on GPU 1 with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` per client: ASR pinned to 30%, TTS 30%, VLM 40% when active. This gives concurrent kernel execution with bounded interference instead of serialized time-slicing.
- **Strict process→GPU pinning** via `CUDA_VISIBLE_DEVICES` in each systemd service unit. No process ever sees both GPUs except the (optional) tensor-parallel configuration below.
- **Alternative topology — tensor parallel:** run one larger model (e.g., a 235B-class MoE at INT4) across both GPUs with vLLM `--tensor-parallel-size 2`. This maximizes reasoning quality but forces perception models onto CPU/iGPU or a third card. **Recommended default is the partitioned layout** — a JARVIS that hears and speaks instantly with a 70–120B brain beats a 235B brain that is deaf while thinking.
- **No NVLink exists on this class of card** — P2P traffic goes over PCIe 5.0 x16 (~64 GB/s/dir). This is why the partitioned (pipeline-parallel-free) layout is preferred: cross-GPU traffic on the hot path is limited to small activation tensors (VLM captions → LLM prompt), which are kilobytes.

### 2.3 Thermal & Power Management

- **Power capping:** `nvidia-smi -pl 480` on both GPUs. Blackwell workstation cards lose only ~5–7% throughput at an 80% power cap while cutting 240 W of heat — the correct trade for a 24/7 appliance.
- **Clock locking:** `nvidia-smi -lgc 1500,2400` to prevent deep idle clock ramps that add ~40–80 ms of first-token jitter after silence. Idle floor costs ~30 W/GPU; latency consistency is worth it.
- **Fan curves:** custom via `nvidia-settings`/IPMI targeting acoustic ceiling of ~42 dBA at the desk; thermal alarm + graceful degrade (drop draft model, cap batch size) at GPU hotspot 90 °C via a `node_exporter` → Prometheus → Alertmanager → orchestrator webhook loop.
- **Storage thermals:** Gen5 NVMe drives *require* the motherboard heatsinks with airflow across them; throttling drives to Gen4 speeds under sustained model loads is a silent failure mode — monitor with `smartctl -a` temperature logging.
- **UPS integration:** NUT (`nut-server`) triggers orchestrator "graceful degrade" mode on battery: checkpoint episodic memory, flush Qdrant WAL, power-cap GPUs to 300 W, then clean shutdown at 20% battery.

---

## 3. The Local AI Stack & Inference Pipeline

### 3.1 Model Selection Matrix (all open-weights, locally deployable)

| Role | Primary Choice | Quantization / Footprint | Alternates | Rationale |
|---|---|---|---|---|
| **Reasoning / Agent LLM** | **GLM-4.5-Air (106B MoE, 12B active)** or **gpt-oss-120b (117B MoE, 5.1B active)** | FP8/MXFP4 → 60–70 GB | Llama-3.3-70B-Instruct (AWQ 4-bit, ~40 GB); Qwen3-32B (fits budget tier); DeepSeek-R1-Distill-Llama-70B for heavy reasoning | MoE architectures give 70B+ quality at 12B-active decode cost → 60–100+ tok/s on one GPU. Strong native tool-calling is mandatory for the agent loop |
| **Vision-Language** | **Qwen3-VL-30B-A3B** | AWQ/FP8, ~24 GB | Qwen2.5-VL-32B-Instruct; Gemma-3-27B (vision); InternVL3-14B | Best open-weights screen understanding: UI element grounding, dense OCR, bounding-box output for click targeting |
| **ASR** | **faster-whisper (Whisper large-v3-turbo, CTranslate2 INT8)** | ~2.5 GB | NVIDIA Parakeet-TDT-0.6B-v2 (English-only, ~3400× realtime, even lower latency); Canary-1B | large-v3-turbo: 4 decoder layers → ~8× realtime decode with near-large-v3 accuracy; multilingual |
| **VAD / Wake** | **Silero VAD v5** + **openWakeWord** | CPU, ~10 MB | Porcupine (commercial) | Runs on CPU at < 1 ms/frame; gates the entire pipeline so GPUs stay idle until speech |
| **TTS (fast path)** | **Kokoro-82M** | FP16, ~0.4 GB | Piper (CPU-only fallback) | ~80–150 ms to first audio chunk; 82M params → negligible VRAM; StyleTTS2-derived quality |
| **TTS (expressive path)** | **XTTS v2** or **Chatterbox / F5-TTS** | ~4 GB | Orpheus-3B (LLM-based, most expressive, higher latency); CosyVoice 2 | Voice cloning + emotional range for long-form responses; streamed sentence-by-sentence |
| **Embeddings** | **Qwen3-Embedding-4B** | FP16, ~8 GB | BGE-M3 (dense+sparse+ColBERT multi-vector in one model); nomic-embed-text-v2 | Top of MTEB among local models; 32K context for long-chunk embedding |
| **Reranker** | **BGE-reranker-v2-m3** | FP16, ~2 GB | Qwen3-Reranker-4B | Cross-encoder rerank of top-50 → top-5; the single highest-ROI RAG quality lever |
| **Router / Utility LLM** | **Qwen3-4B-Instruct** | FP8, ~4 GB | Llama-3.2-3B | Intent classification, query rewriting, memory summarization, PII redaction — never burn 70B tokens on plumbing |
| **Code specialist (optional hot-swap)** | **Qwen3-Coder-30B-A3B** | AWQ | Devstral-Small | Loaded on demand into GPU 1 headroom for heavy code-agent sessions |

### 3.2 Inference Engine Selection

| Engine | Use in this system | Why / Why not |
|---|---|---|
| **vLLM (V1 engine)** | **Primary serving for LLM + VLM** (one instance per GPU) | PagedAttention (near-zero KV fragmentation), continuous batching (agent tool-calls + user chat interleave without head-of-line blocking), **prefix caching** (system prompt + RAG preamble cached → TTFT drops 60–80%), speculative decoding with draft model, FP8 KV cache, OpenAI-compatible API, native tool-call parsing |
| **SGLang** | Alternate to vLLM; adopt if agent workloads dominate | RadixAttention prefix-tree caching is superior for agentic loops that repeatedly re-submit near-identical long prompts; measure both, keep one |
| **TensorRT-LLM** | Optional optimization pass for the *frozen* production model | Highest absolute tok/s (10–25% over vLLM) but hours-long engine builds per model/quant/context combo; adopt only after the model choice is stable |
| **llama.cpp** | **Degraded-mode fallback + utility models on CPU** | GGUF + partial CPU offload means the assistant survives a GPU failure at reduced speed (8-channel DDR5 sustains ~8–12 tok/s on a 70B Q4); also serves the router model with zero GPU cost if desired |
| **CTranslate2** | ASR runtime (inside faster-whisper) | INT8 Whisper kernels, best-in-class ASR throughput |
| **ExLlamaV2/V3** | Not used | Excellent single-user tok/s but weaker concurrent-request semantics than vLLM for the agent+chat mix |

**Serving layout (systemd units):**

```
vllm-primary.service   → GPU0: GLM-4.5-Air FP8, --enable-prefix-caching
                          --speculative-model Qwen3-4B --max-model-len 131072
                          --kv-cache-dtype fp8, listens on unix:/run/jarvis/llm.sock
vllm-vision.service    → GPU1: Qwen3-VL-30B AWQ, --max-model-len 32768
asr.service            → GPU1: faster-whisper large-v3-turbo INT8 (MPS 30%)
tts.service            → GPU1: Kokoro + XTTS v2 (MPS 30%)
embed.service          → GPU1: Qwen3-Embedding + BGE-reranker (via TEI or infinity)
router.service         → GPU1: Qwen3-4B FP8 (vLLM, small footprint)
```

### 3.3 The Real-Time Speech Pipeline (Audio → ASR → LLM → TTS)

```
 Mic (48 kHz) ──► [PipeWire] ──► Ring buffer (20 ms frames)
                                      │
                              [Silero VAD v5]  ◄── openWakeWord ("Jarvis")
                                      │  speech gate + endpointing
                                      ▼
                     [faster-whisper large-v3-turbo]
                     streaming: 1 s windows, LocalAgreement-2
                     partial transcripts every ~200 ms
                                      │
                     ┌────────────────┴───────────────┐
                     ▼ (partials)                     ▼ (final, on endpoint)
             [Router: Qwen3-4B]                [Orchestrator]
             intent pre-classification         prompt assembly:
             + speculative RAG prefetch        system + memory + RAG + tools
                                      │
                                      ▼
                     [vLLM / GLM-4.5-Air — streaming tokens]
                                      │
                     [Sentence segmenter: emit on . ! ? , ≥ 12 tokens]
                                      │
                     ┌────────────────┴───────────────┐
                     ▼                                ▼
             [Kokoro TTS]                     [Tool-call branch]
             per-clause synthesis,            → broker → sandbox → result
             24 kHz PCM chunks                → re-enter LLM loop
                     │
                     ▼
             [PipeWire playback]  ◄── barge-in: VAD speech during playback
                                      cancels TTS + LLM generation (hard flush)
```

Key engineering decisions:

- **Streaming ASR with LocalAgreement:** run Whisper on a sliding 1 s window; commit tokens only when two consecutive windows agree (whisper-streaming technique). Gives stable partials at ~200 ms lag instead of waiting for utterance end.
- **Speculative prefetch:** the moment the router classifies a partial transcript as "knowledge question," embedding + vector search fire *before the user stops talking*. RAG results are ready when the final transcript lands — retrieval latency is hidden entirely.
- **Clause-level TTS chunking:** the LLM's token stream is segmented at clause boundaries and shipped to Kokoro immediately. First audio plays while the LLM is still generating sentence three. Cross-fade 10 ms between chunks to hide seams.
- **Barge-in (full duplex):** VAD runs continuously during playback with echo cancellation (PipeWire's `module-echo-cancel`, WebRTC AEC). User speech ≥ 300 ms cancels generation via vLLM's abort API and flushes the audio queue. This one feature is 80% of what makes it feel "alive."

### 3.4 Concurrency Model

- The orchestrator is a single **Python 3.12 asyncio** process (uvloop) — all I/O-bound. CPU-bound audio DSP lives in a separate process connected by shared-memory ring buffers (`multiprocessing.shared_memory`), never in the event loop.
- vLLM's continuous batching means a background agent task (e.g., "summarize today's logs") and a live voice exchange share GPU 0 without either blocking; live voice requests are tagged with vLLM priority scheduling.

### 3.5 End-to-End Latency Budget (voice-to-voice)

| Stage | Budget | Technique |
|---|---|---|
| Audio capture + VAD endpoint detection | 120 ms | 20 ms frames, aggressive endpointing (300 ms trailing silence) |
| ASR final transcript | 90 ms | Partials already computed; endpoint just commits |
| RAG retrieval + rerank | 0 ms *(effective)* | Speculatively prefetched during speech; else 45 ms budget |
| Prompt assembly + tokenization | 15 ms | Pre-tokenized system prompt, prefix cache hit |
| LLM time-to-first-token | 180 ms | Prefix caching + FP8 + speculative decoding |
| First clause complete (~15 tokens) | 150 ms | ~100 tok/s decode |
| TTS first audio chunk | 90 ms | Kokoro, clause-level |
| Playback buffer | 40 ms | PipeWire quantum 1024 @ 24 kHz |
| **Total (P50)** | **~685 ms** | **Target: < 800 ms P90** |

---

## 4. Multi-Tiered Memory & RAG Architecture

### 4.1 The Four-Tier Memory Hierarchy

| Tier | Store | Latency | Contents | Eviction / Consolidation |
|---|---|---|---|---|
| **T0 — In-flight** | vLLM paged KV cache + prefix cache | µs | Current conversation tokens, cached system prompt + persona + tool schemas | LRU within vLLM; conversation > 100K tokens triggers rolling summarization by router model |
| **T1 — Ephemeral session** | **Redis 7** (unix socket, AOF off) | < 1 ms | Active window state, last N screen captions, pending tool results, scratchpad, TTL'd speculative RAG prefetches | TTL 15–120 min; lost on reboot by design |
| **T2 — Episodic** | **SQLite (WAL mode) + FTS5**, one DB per month | < 5 ms | Every interaction: `(ts, speaker, text, intent, tools_used, outcome, app_context, embedding_id)`. Task ledger: goals, plans, completions, failures | Never deleted; nightly consolidation summarizes + promotes to T3 |
| **T3 — Semantic (KB)** | **Qdrant** (on-disk HNSW, mmap) + **Tantivy** BM25 index + **MinIO** blob store | 10–40 ms | Vectorized documents, code, past-conversation summaries, extracted facts/preferences, entity graph | Re-embedding on model upgrade via versioned collections; tombstoned on user request ("forget X") |

**The consolidation daemon ("hippocampus"):** a nightly (and idle-triggered) job using the router model that:
1. Replays the day's episodic log and extracts durable facts ("user's staging server is `10.0.4.20`", "prefers pytest over unittest") into T3 with `source_episode` provenance links.
2. Writes a 200-token "day summary" embedding for temporal queries ("what was I working on last Tuesday?").
3. Detects contradictions with existing T3 facts and resolves by recency + confidence, keeping superseded facts with `valid_until` timestamps (bitemporal memory — the system can answer "what did I *used to* prefer?").
4. Decays retrieval weight of unaccessed episodic summaries (exponential, half-life 90 days) rather than deleting.

### 4.2 Ingestion Pipeline (Documents, Code, Logs)

```
watchdog (inotify) on ~/Documents, ~/Projects, /var/log/jarvis-scoped
        │
        ▼
[Extractor] ── PDFs: PyMuPDF + Docling (tables/layout) ── Office: Docling
              ── Code: tree-sitter AST parse ── HTML: trafilatura
        │
        ▼
[Chunker — type-specific, see 4.3]
        │
        ▼
[Enricher] router-model generates: title, 1-line summary, keywords,
           + contextual header ("This chunk is from §3 of X, which covers Y")
           (Anthropic-style contextual retrieval — cuts failed retrievals ~50%)
        │
        ▼
[Embedder] Qwen3-Embedding-4B (dense, 2560-d) + BM25 tokens → Tantivy
        │
        ▼
[Qdrant upsert] collection per domain: docs / code / episodic / facts / screen
```

### 4.3 Chunking Strategy (per content type)

| Content | Strategy | Size / Overlap |
|---|---|---|
| Prose / PDFs | Semantic chunking: split at embedding-similarity valleys between sentences; respect heading boundaries from Docling layout | 400–800 tokens, 15% overlap |
| **Code** | **AST-aware via tree-sitter**: one chunk per function/class, with file path, imports, and enclosing-scope signature prepended as context header. Never split a function mid-body | ≤ 1200 tokens; large functions get sliding windows *with* signature repeated |
| Logs | Time-window + template mining (Drain3 to cluster log templates); store templates, not raw repetition | 5-min windows |
| Conversations | One chunk per exchange + daily summary chunk | natural |
| Screen captures | VLM caption + OCR text per capture event, keyed to `(app, window_title, ts)` | event-based |

### 4.4 Retrieval Path (the hot query pipeline)

1. **Query rewrite** (router model, 30 ms): resolve pronouns/ellipsis from conversation ("fix *that* function" → concrete symbol name); generate 1–2 paraphrases (multi-query).
2. **Hybrid search, parallel:** Qdrant dense top-50 (HNSW `ef=128`) + Tantivy BM25 top-50, fused with **Reciprocal Rank Fusion** (k=60). Dense catches paraphrase; BM25 catches exact identifiers, error codes, IPs — non-negotiable for code and logs.
3. **Metadata pre-filter:** Qdrant payload filters on `domain`, `mtime`, `project`, `acl_tier` *before* ANN (filterable HNSW) — e.g., "recent" queries filter `mtime > now-7d` instead of hoping recency emerges from similarity.
4. **Rerank:** BGE-reranker-v2-m3 cross-encodes top-50 → top-5 (~25 ms on GPU 1).
5. **Assembly with provenance:** each chunk enters the prompt wrapped in `<doc id="…" source="path#L120-L162" mtime="…">`; the system prompt mandates citation of `id`s, and the UI renders them as clickable links. **Answers without a retrievable source for factual claims are instructed to say so** — the primary anti-hallucination control, enforced by a post-hoc citation-checker pass (router model verifies each cited id exists and is topical).
6. **Score floor:** if the reranker's top score < threshold, retrieval is declared *failed* and the LLM is told "no relevant local knowledge found" — silence beats plausible garbage.

**Qdrant tuning:** scalar INT8 quantization with `always_ram: true` for quantized vectors + mmap originals (rescoring on); HNSW `m=16, ef_construct=256`; separate collections per domain so HNSW graphs stay small and per-domain snapshots are cheap.

---

## 5. Agentic Orchestration & OS-Level Tooling

### 5.1 Process Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ jarvis-orchestrator (Python 3.12 / asyncio, systemd)         │
│   • LangGraph state machine (plan → act → observe → verify)  │
│   • Session/turn management, barge-in control                │
│   • Talks to everything via gRPC over unix domain sockets    │
├──────────────────────────────────────────────────────────────┤
│ jarvis-perception (Rust)          jarvis-broker (Rust)       │
│   • screen capture, window/       • THE ONLY path to side    │
│     input tracking, hotkeys         effects; policy engine   │
│   • publishes on ZeroMQ pub/sub     (Cedar/OPA) + audit log  │
├──────────────────────────────────────────────────────────────┤
│ Sandboxed runners (spawned per task by broker)               │
│   • code-runner: Docker + gVisor (runsc)                     │
│   • shell-runner: bubblewrap, read-only rootfs               │
│   • browser-runner: Playwright in container                  │
│   • gui-runner: ydotool/uinput (Linux) — direct, policy-gated│
├──────────────────────────────────────────────────────────────┤
│ UI: Tauri desktop app  ◄── WebSocket ──►  orchestrator       │
│   overlay HUD, transcript, citation links, kill switch       │
└──────────────────────────────────────────────────────────────┘
```

- **Transport:** gRPC (proto3) over **unix domain sockets** for all service-to-service calls — no TCP on localhost, so nothing to firewall internally and ~30% lower latency than loopback TCP. High-rate perception events (60 Hz window/cursor state) go over **ZeroMQ pub/sub** with msgpack framing. The UI gets a single WebSocket multiplexing transcript, audio, and state-diff streams.
- **Tool protocol:** all tools are exposed as **MCP (Model Context Protocol) servers** registered with the orchestrator. This gives a uniform schema (JSON Schema tool definitions), lets you drop in the growing ecosystem of MCP servers (filesystem, git, browsers), and keeps tool definitions decoupled from the agent loop. Custom tools are FastAPI/gRPC services wrapped in a thin MCP adapter.

### 5.2 Perception: What the Assistant Senses

| Stream | Linux (primary target) | Windows equivalent | Rate / Notes |
|---|---|---|---|
| Screen pixels | **PipeWire screen-cast via XDG desktop portal** (Wayland-native, user-consented) | Windows.Graphics.Capture (WinRT) | On-demand + on-window-change; frames → VLM only when needed. Continuous 1 fps "ambient" captioning is opt-in |
| Active window / focus | `wlr-foreign-toplevel` protocol or KWin scripting API; X11 fallback: `libxcb` EWMH | `GetForegroundWindow` + UIAutomation | Event-driven; feeds T1 memory `(app, title, ts)` |
| Accessibility tree | **AT-SPI2** (`pyatspi`) — text + roles + bounding boxes of UI elements | UIA (`pywinauto`/`uiautomation`) | *Preferred over pixels*: 100× cheaper than VLM and exact. VLM is the fallback for canvas-rendered apps |
| Cursor / input cadence | libinput event stream (read-only) | Low-level hooks | Used for "is the user busy?" presence model; raw keystrokes are **never** logged — only activity envelopes |
| Clipboard | `wl-clipboard` watcher (opt-in) | Win32 clipboard listener | Explicit toggle; contents TTL 15 min in T1 |
| Audio | PipeWire graph as in §3.3 | WASAPI | Always-on VAD, GPU wakes on speech |
| System telemetry | eBPF-lite: `psutil` + `nvidia-smi dmon` + journald tail into episodic log | ETW | Lets the agent answer "why is the fan loud?" with real data |

**Screen-to-action grounding:** for GUI automation, the VLM (Qwen3-VL grounding mode) returns normalized bounding boxes for targets ("the Save button"); the gui-runner converts to absolute coordinates and issues `uinput` events, then *verifies* the post-action screenshot matches the expected state before reporting success (act → observe → verify, never fire-and-forget).

### 5.3 The Agent Loop (LangGraph)

A typed state machine, not a free-running ReAct loop:

```
   [intake] → [route] ──single-shot──► [respond]
                │
                └─agentic──► [plan] → [act_k] → [observe_k] → [verify]
                               ▲                                │
                               └──── revise (max 8 iterations) ─┘
                                                │
                                     [checkpoint → respond]
```

- **Route:** router model classifies: `chat` (no tools) / `retrieval` / `agentic` / `gui-automation`. 70% of turns never enter the agent loop — this is the main tok/s saver.
- **Plan:** primary LLM emits a structured plan (Pydantic-validated JSON): steps, tools, expected artifacts, **risk tier per step**.
- **Act:** tool calls stream through the broker (§5.4). Parallel tool fan-out where the plan's dependency graph allows.
- **Verify:** a *separate* LLM pass checks artifacts against the plan's acceptance criteria (tests pass? file exists? screenshot shows expected state?). Failed verification → revise, with the failure appended. Hard ceilings: 8 iterations, 5 min wall-clock, per-task token budget — then escalate to the user with a checkpointed state that can be resumed.
- **Checkpointing:** LangGraph's SQLite checkpointer persists agent state; a crashed or interrupted task resumes instead of restarting.

### 5.4 The Broker: Policy-Gated Execution

Every side effect is a gRPC call to `jarvis-broker`, which:

1. **Authenticates** the caller (per-service mTLS on unix sockets via SO_PEERCRED).
2. **Evaluates policy** (Cedar policy language): tool × argument-pattern × risk tier × current mode.
3. **Executes in the matching sandbox** and returns structured results + captures stdout/stderr/artifacts to the audit log.

| Risk tier | Examples | Policy |
|---|---|---|
| T0 read-only | read file in whitelisted dirs, list windows, search KB | Auto-allow, logged |
| T1 reversible write | write in `~/jarvis-workspace`, run code in sandbox, git commit to branch | Auto-allow, logged, snapshot-first (btrfs/ZFS snapshot of workspace) |
| T2 system-touching | install package, edit dotfiles, GUI automation on real apps, send network request (if egress window open) | Require **spoken/clicked confirmation** with plain-language diff preview |
| T3 destructive/irreversible | delete outside workspace, `sudo`, disk ops, anything touching credentials | Denied by default; explicit per-invocation unlock (hardware key tap — YubiKey touch) |

**Sandbox implementations:**
- **code-runner:** Docker with **gVisor (runsc)** runtime, `--network none` by default, seccomp default profile, read-only bind of requested inputs, tmpfs workdir, 4 CPU / 8 GB / 120 s ceilings, artifacts copied out through the broker. Firecracker microVMs are the upgrade path if untrusted-code volume grows.
- **shell-runner:** bubblewrap (`bwrap`) — instant startup (< 10 ms vs ~300 ms for Docker) for T0/T1 commands; new user+mount+pid namespaces, read-only `/`, whitelisted binds.
- **browser-runner:** Playwright + Chromium inside the container; downloads land in a quarantine dir scanned before promotion.
- **Prompt-injection containment:** any content that entered the context from external sources (web pages, downloaded files, OCR of untrusted windows) is wrapped in delimited untrusted blocks, and the broker *drops the agent to T0 permissions for the remainder of any turn whose context contains untrusted content* unless the user confirms. Tool results are data, never instructions — enforced by system prompt *and* by the tier drop, because prompts alone don't survive adversarial input.

---

## 6. Security Hardening & Data Privacy Blueprint

### 6.1 Threat Model (explicit)

In scope: exfiltration of personal data by malicious documents/web content (prompt injection), supply-chain compromise of model weights or Python deps, lateral movement from a sandboxed tool, physical theft of the node, and the assistant itself taking unsafe actions. Out of scope: nation-state physical attacks, side-channels against the GPU.

### 6.2 Data-at-Rest & Boot Integrity

- **Full-disk encryption:** LUKS2 (aes-xts-plain64) on all data arrays, keys sealed to **TPM 2.0 + PIN** (`systemd-cryptenroll --tpm2-with-pin=yes`); Secure Boot with custom MOK; kernel lockdown mode `integrity`.
- **Model provenance:** weights are pulled once, hash-pinned (SHA-256 manifest, `safetensors` only — **never** pickle-format checkpoints), and verified at service start. Python env is uv-locked with hash checking (`--require-hashes`); dependency updates only in explicit maintenance windows, reviewed via `pip-audit`/osv-scanner.
- **Secrets:** no plaintext API keys anywhere. `systemd-creds` (TPM-sealed) for service credentials; user secrets in a local `pass`/age store; the LLM never sees raw secrets — the broker injects them into tool execution environments and **redacts them (plus regex/entropy scans) from all tool output** before it re-enters model context.

### 6.3 Network Posture: Default-Deny, Provable

```nftables
table inet jarvis {
  chain output {
    type filter hook output priority 0; policy drop;
    oifname "lo" accept
    meta skuid "jarvis-failover" ct state new tcp dport 443 \
        ip daddr @allowed_endpoints accept   # only during open windows
    counter log prefix "jarvis-egress-blocked: " drop
  }
}
```

- **Policy DROP on egress** for every uid except a dedicated `jarvis-failover` user, whose allowed endpoint set is empty except during explicitly opened windows. All AI services run as uids with *zero* network capability — a fully compromised inference process cannot phone home.
- All inter-service traffic on unix sockets → there are no listening TCP ports at all (`ss -tlnp` returns nothing but the UI's localhost WebSocket, bound to 127.0.0.1 with an auth token).
- **Update windows:** model/dependency updates happen in a scheduled window where nftables loads a temporary allowlist (HuggingFace + distro mirrors, IP-pinned), with all traffic through a logging proxy (mitmproxy in transparent mode) whose transcript is archived.
- Blocked-egress counters are surfaced on the HUD — an exfiltration *attempt* (e.g., from injected content convincing a tool to `curl`) becomes a visible security event, not a silent drop.

### 6.4 Audit & Accountability

- **Append-only audit log:** every broker decision `(ts, caller, tool, args-hash, policy-verdict, result-hash)` written to a local ledger with per-entry hash chaining (Merkle-style), synced to the cold tier; tamper-evidence via daily root-hash printed to the HUD and optionally to a hardware token.
- **Flight recorder:** rolling 24 h of full pipeline traces (OpenTelemetry → local Jaeger) — every prompt, retrieval set, tool call, and latency span, queryable when something behaves oddly. Retention configurable; encrypted at rest like everything else.
- **Kill switches:** (1) global hotkey + HUD button → SIGSTOP all runners, abort vLLM generations, mute mic; (2) hardware mic mute; (3) `systemctl stop jarvis.target` stops the world. Muted state is reflected by a physical LED (USB busylight) that the software cannot fake-off.

### 6.5 Privacy Engineering

- **Data minimization at capture:** keystroke contents never logged; screen captures processed → captioned → *pixels discarded* by default (captions retained, frames only kept if user pins them); clipboard and ambient-capture are opt-in toggles with HUD indicators.
- **Right-to-forget:** "Jarvis, forget everything about X" → router model resolves X to entities → tombstones in Qdrant + episodic rows redacted (crypto-shredding: per-entity envelope keys where feasible) → confirmation lists what was removed.
- **Local telemetry only.** No usage analytics leave the machine. Ever.

### 6.6 Graceful Degradation & Edge Failover Ladder

| Level | Trigger | Behavior |
|---|---|---|
| **L0 — Full** | Normal | Everything in §2–§5 |
| **L1 — Thermal/power degrade** | GPU > 90 °C, UPS on battery | Drop speculative decoding, cap context 32K, power-cap 300 W, defer background jobs |
| **L2 — Single-GPU survival** | GPU failure | Perception stack migrates to remaining GPU at reduced quality (Whisper-small, Kokoro only, VLM off); primary LLM swaps to Qwen3-32B AWQ from the RAID-0 model store (< 30 s) |
| **L3 — CPU lifeboat** | Both GPUs down | llama.cpp: Qwen3-30B-A3B GGUF Q4 on 64 Zen 5 cores + 409 GB/s DDR5 (~20–30 tok/s MoE); Piper CPU TTS; whisper.cpp small. Slow, but *never dark* |
| **L4 — Cloud escalation (opt-in, off by default)** | User explicitly requests, or task exceeds local capability *and* policy allows | Router drafts an escalation request → **PII/secret redaction pass** (Presidio + custom recognizers over the entire outbound payload, diff shown to user) → user confirms → egress window opens to a single pinned endpoint (your own GPU box over WireGuard/Tailscale, or a commercial API) → response returns → window closes. Sensitive-tagged memory tiers are *structurally excluded* from escalation payload assembly — not filtered, excluded: the escalation builder has no read path to `acl_tier: private` collections |

---

## Appendix A — Software Bill of Materials (pinned stack)

| Layer | Selection |
|---|---|
| OS | Ubuntu Server 24.04 LTS (HWE kernel) + PipeWire; GNOME on Wayland if used as a daily driver |
| GPU stack | NVIDIA driver 5xx, CUDA 12.x, MPS enabled on GPU 1 |
| Serving | vLLM ≥ 0.8 (V1), CTranslate2/faster-whisper, text-embeddings-inference or infinity |
| Orchestration | Python 3.12, uv, LangGraph, Pydantic v2, grpcio, pyzmq, MCP SDK |
| Perception/broker | Rust (tokio, tonic, zeromq), Cedar policy engine |
| Memory | Qdrant ≥ 1.12, Tantivy, Redis 7, SQLite 3.45 (WAL), MinIO, Drain3, Docling, tree-sitter |
| Sandbox | Docker + gVisor runsc, bubblewrap, Playwright |
| UI | Tauri 2 (Rust + TypeScript), WebSocket, PCM streaming via Web Audio |
| Ops | systemd units + `jarvis.target`, Prometheus + node_exporter + dcgm-exporter, Grafana, Alertmanager, OpenTelemetry + Jaeger, NUT, restic (cold tier), btrfs/ZFS snapshots |

## Appendix B — Build-Out Order (pragmatic sequencing)

1. **Week 1–2:** Node assembly, OS hardening (§6.2–6.3), storage arrays, systemd skeleton, monitoring.
2. **Week 3–4:** vLLM primary LLM + speech pipeline (§3.3) → hit the < 800 ms voice loop *before anything else*. This is the foundation everything is judged against.
3. **Week 5–6:** Memory plane: ingestion, Qdrant/Tantivy hybrid retrieval, reranking, citations (§4).
4. **Week 7–9:** Broker + sandboxes + first 10 tools (files, shell, code-runner, browser); LangGraph loop with verify step (§5).
5. **Week 10+:** Perception streams, GUI grounding, consolidation daemon, failover ladder drills (deliberately pull a GPU; unplug the NIC; kill -9 the orchestrator mid-task and watch it resume).

---

*End of specification.*
