# KServe Analysis: Multi-Runtime Support in LLMInferenceService (SGLang & TensorRT-LLM)

* **Author:** Mohan Kumar
* **Status:** Analysis / Discovery
* **Target Version:** v0.17+
* **Companion Doc:** [multi_runtime_support_required_changes.md](./multi_runtime_support_required_changes.md)

---

## 1. Overview & Problem Statement

KServe's `LLMInferenceService` (LLMISVC, `serving.kserve.io/v1alpha2`) is becoming our standard control plane for serving large language models on Kubernetes. Today it is implemented and tested almost exclusively around **vLLM** (packaged inside the `ghcr.io/llm-d/llm-d-cuda` images) and the **llm-d** routing stack (Endpoint Picker / EPP, routing sidecar, tokenizer) layered on top of the **Gateway API Inference Extension (GIE)**.

To position KServe as a neutral, runtime-agnostic serving platform, we want first-class support for the other two dominant open inference engines:

1. **SGLang** (`sgl-project/sglang`) — high-throughput engine with RadixAttention prefix caching, strong P/D disaggregation, and broad hardware coverage.
2. **NVIDIA TensorRT-LLM** (`NVIDIA/TensorRT-LLM`) — NVIDIA's optimized engine, now shipping a modern OpenAI-compatible server (`trtllm-serve`) with a PyTorch backend.

This document answers four questions the team raised:

1. **What changes are required** to integrate SGLang and TensorRT-LLM with `LLMInferenceService`? *(High-level here; mechanics in the companion doc.)*
2. **Can the llm-d features be driven purely by metrics scraping** (rather than runtime-specific code)?
3. **Does llm-d publish native images** for SGLang and TensorRT-LLM the way it does for vLLM?
4. **What is the realistic feature gap** per runtime (single-node, multi-node, P/D disaggregation, prefix-cache routing, LoRA)?

> [!NOTE]
> **Bottom line up front:** The architecture is *structurally* runtime-agnostic — the controller delegates the full `PodSpec` to the user and the GIE routing layer talks to backends over the OpenAI HTTP API plus standard Prometheus metrics. **SGLang is already an officially supported llm-d engine (since llm-d v0.6.0).** **TensorRT-LLM is recognized in the GIE metric protocol but is not yet a tested path in llm-d** (tracked upstream in [llm-d#1550](https://github.com/llm-d/llm-d/issues/1550)). The work for KServe is therefore mostly **new preset `LLMInferenceServiceConfig` objects + a small amount of controller plumbing (engine-type label, port/probe configurability)** — not a rewrite.

---

## 2. The Runtime Landscape

| Capability | vLLM (today) | SGLang | TensorRT-LLM |
|---|---|---|---|
| OpenAI server entrypoint | `vllm serve` | `python -m sglang.launch_server` | `trtllm-serve` |
| Default port | 8000 | **30000** | 8000 |
| OpenAI endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) | ✅ | ✅ | ✅ |
| Health endpoint | `/health` | `/health` (live), **`/health_generate`** (ready) | `/health` (Triton: `/v2/health/ready`) |
| Prometheus `/metrics` | ✅ always | ✅ via `--enable-metrics` | ⚠️ weak on bare `trtllm-serve`; full via Triton (`:8002`) or Dynamo |
| Loads HF checkpoint directly | ✅ | ✅ | ✅ **with `--backend pytorch`** (no `trtllm-build`) |
| Official image | `vllm/vllm-openai`, `ghcr.io/llm-d/llm-d-*` | `lmsysorg/sglang` | `nvcr.io/nvidia/tensorrt-llm/release`, `nvcr.io/nvidia/tritonserver` |
| Multi-node | LWS + Ray/native | `--nnodes/--node-rank/--dist-init-addr` | MPI/`mpirun` + LWS (Triton leader/worker) |
| P/D disaggregation | NIXL connector | Mooncake **or** NIXL (`--disaggregation-mode`) | `trtllm-serve disaggregated` (NIXL/UCX/MPI) |
| Prefix-cache (KV-event) routing | ✅ (ZMQ events) | ✅ (ZMQ events) | Protocol-aware, not wired in llm-d |
| LoRA dynamic serving | ✅ | ✅ (`--lora-paths`, `--max-loras-per-batch`) | Limited / engine-build bound |

**Key takeaway:** SGLang and `trtllm-serve` (PyTorch backend) both behave like vLLM at the contract level — point them at an HF model, they expose OpenAI endpoints and a health probe. The differences are in **flag names, ports, probe paths, metric names, and KV-transfer connectors** — exactly the things KServe currently hardcodes for vLLM.

---

## 3. How `LLMInferenceService` Works Today (and Where vLLM Is Baked In)

### 3.1 Template delegation — the runtime-agnostic core

The controller does **not** synthesize containers from scratch. It copies the user's (or a preset's) `PodSpec` verbatim into the generated `Deployment`/`LeaderWorkerSet`:

```go
// pkg/controller/v1alpha2/llmisvc/workload_single_node.go (and workload_multi_node.go)
d.Spec.Template.Spec = *llmSvc.Spec.Template.DeepCopy()
```

The API surface that feeds this ([`llm_inference_service_types.go`](file:///Users/mohankumar/kluisz/kserve/pkg/apis/serving/v1alpha2/llm_inference_service_types.go)):

- `spec.template` — main/decode/single-node leader `PodSpec`
- `spec.worker` — multi-node worker `PodSpec` (triggers a `LeaderWorkerSet`)
- `spec.prefill` — disaggregated prefill workload
- `spec.model` — model URI + name + LoRA spec
- `spec.parallelism` — tensor/pipeline/data parallel, expert parallel
- `spec.router` — `{ gateway, route, scheduler }` (the GIE/EPP layer)
- `spec.baseRefs` — references to `LLMInferenceServiceConfig` presets that get merged in

**Implication:** A user *can* run SGLang or TensorRT-LLM **today** by supplying a complete `spec.template` with the right image, command, port, and probes — no code change required. This is the same property that made the [NVIDIA DRA proposal](./nvidia_dra_support_proposal.md) a no-code-change feature. What's missing is the *ergonomic* path: presets, defaults, and the routing-layer glue.

### 3.2 The presets are vLLM-only

The merge logic ([`config_merge.go`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/config_merge.go)) auto-applies "well-known" `LLMInferenceServiceConfig` presets based on the deployment shape. **Every shipped preset hardcodes vLLM:**

| Preset (`config/llmisvcconfig/`) | vLLM assumption |
|---|---|
| `config-llm-template.yaml` | `image: ghcr.io/llm-d/llm-d-cuda:v0.6.0`, `exec vllm serve /mnt/models --port 8000`, `containerPort: 8000`, probes on `/health` |
| `config-llm-decode-template.yaml` | vLLM + `ghcr.io/llm-d/llm-d-routing-sidecar` on 8000, decode on 8001 |
| `config-llm-prefill-template.yaml` | vLLM prefill on 8000 |
| `config-llm-worker-data-parallel.yaml` | `--data-parallel-size`, `--data-parallel-address`, `--data-parallel-rpc-port 5555`, RoCE inference shell |
| `config-llm-scheduler.yaml` | EPP `ghcr.io/llm-d/llm-d-inference-scheduler` + `llm-d-uds-tokenizer`, `defaultEngine: vllm` |

### 3.3 vLLM-specific assumptions inventory

| Assumption | Location | Notes |
|---|---|---|
| Workload Service port **8000** (hardcoded, deliberately) | [`workload.go:131-148`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/workload.go) | Comment explicitly says changing it "requires a lot of changes beyond this service (presets, routing sidecar flags…)". SGLang defaults to **30000**. |
| `vllm serve` command + flags | all `config-llm-*` presets | SGLang/TRT-LLM use different binaries and flag spellings |
| `/health` probe path | all presets | SGLang readiness is best served by `/health_generate`; Triton uses `/v2/health/ready` |
| `--tensor-parallel-size`, `--data-parallel-size`, `--data-parallel-rpc-port` | worker presets | map from `spec.parallelism.*`; flag names differ per engine |
| LoRA flags `--max-lora-rank/--max-loras/--max-cpu-loras` | presets | SGLang: `--max-lora-rank/--max-loras-per-batch/--lora-paths` |
| EPP metric plugin `defaultEngine: "vllm"` | [`scheduler.go:1510-1529`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/scheduler.go) | See §4 — already engine-aware via a label, just defaults to vLLM |
| Routing sidecar connector (NIXL) | decode presets | SGLang needs `--connector=sglang`; TRT-LLM has no llm-d connector |

---

## 4. Can the llm-d Features Be Driven by Metrics Scraping? — **Yes**

This is the most important architectural finding. The llm-d routing intelligence (load-aware, queue-aware, KV-cache-aware scheduling) is built on the **Gateway API Inference Extension (GIE) "Model Server Protocol"** ([proposal 003](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/docs/proposals/003-model-server-protocol/README.md)), which is **explicitly engine-agnostic**:

> *"The model server MUST implement OpenAI's Completions and Chat APIs"* and expose three core Prometheus gauges. *"The exact metric names don't necessarily need to be the same as the recommended names here; however the metric types and semantics MUST follow this doc."*

So the EPP doesn't *care* that vLLM emits `vllm:num_requests_waiting` — it cares that **some** gauge represents the queue depth. The mapping from semantic → concrete metric name is **configuration, selected by a pod label.**

### 4.1 The mapping is already in KServe's controller

KServe's scheduler reconciler already injects the GIE core-metrics plugin and is **already engine-aware** — it just defaults to vLLM:

```go
// pkg/controller/v1alpha2/llmisvc/scheduler.go (≈1510-1529)
pluginEntry := map[string]interface{}{
    "name": coreMetricsExtractorPlugin,           // "model-server-protocol-metrics"
    "type": coreMetricsExtractorPlugin,
    "parameters": map[string]interface{}{
        "engineLabelKey": "inference.networking.k8s.io/engine-type",
        "defaultEngine":  "vllm",                  // <-- the only thing pinning us to vLLM
        "engineConfigs":  []interface{}{ engineConfig },
    },
}
```

The fields it understands (`scheduler.go:1484-1490`) map 1:1 to the GIE per-engine spec:

| Semantic | engineConfig field |
|---|---|
| Total queued requests | `queuedRequestsSpec` |
| Total running requests | `runningRequestsSpec` |
| KV-cache utilization | `kvUsageSpec` |
| LoRA info | `loraSpec` |
| Cache info (blocks/size) | `cacheInfoSpec` |

### 4.2 The per-runtime metric map (from the GIE/llm-d protocol)

| Semantic (GIE) | vLLM | SGLang | trtllm-serve | Triton TRT-LLM |
|---|---|---|---|---|
| Queued requests | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | `trtllm_num_requests_waiting` | `nv_trt_llm_request_metrics{request_type=waiting}` |
| Running requests | `vllm:num_requests_running` | `sglang:num_running_reqs` | `trtllm_num_requests_running` | `nv_trt_llm_request_metrics{request_type=scheduled}` |
| KV-cache utilization | `vllm:kv_cache_usage_perc` | `sglang:token_usage` | `trtllm_kv_cache_utilization` | `nv_trt_llm_kv_cache_block_metrics{...=fraction}` |

llm-d selects which row applies via the pod label **`llm-d.ai/engine-type: {vllm|sglang|trtllm-serve|triton-tensorrt-llm}`** (pods with no label default to vLLM). KServe's controller uses the equivalent `inference.networking.k8s.io/engine-type` key.

> [!IMPORTANT]
> **Conclusion:** Queue-aware, load-aware, and KV-cache-utilization routing — the bulk of llm-d's "smart routing" value — **work for SGLang and TensorRT-LLM purely through metrics scraping + the right engine-type label + the right metric-name config.** No EPP code changes are needed for these features. The only routing features that are *genuinely* engine-coupled are:
> - **LoRA-affinity scoring** — the algorithm and `vllm:lora_requests_info` format are biased to vLLM's dynamic LoRA; not portable to SGLang/TRT-LLM today.
> - **Precise prefix-cache routing** — needs the engine to publish KV-block events over ZMQ. vLLM and SGLang do; TensorRT-LLM does not in the llm-d wiring.
> - **P/D routing sidecar** — needs a per-engine KV-transfer *connector* (vLLM→`nixlv2`, SGLang→`sglang`, TRT-LLM→none).

---

## 5. Does llm-d Publish Native Images for SGLang / TensorRT-LLM? — **No**

llm-d's custom-built model-server images are **vLLM-only**. There is **no `llm-d-sglang` or `llm-d-tensorrt-llm` image.** ([llm-d artifacts doc](https://github.com/llm-d/llm-d/blob/main/docs/getting-started/artifacts.md))

**llm-d-built model-server images (all vLLM):**

| Image | Target |
|---|---|
| `ghcr.io/llm-d/llm-d-cuda` | NVIDIA GPU (vLLM) |
| `ghcr.io/llm-d/llm-d-cuda-gb200` | GB200 / DeepEP |
| `ghcr.io/llm-d/llm-d-aws` | NVIDIA + EFA/libfabric + NIXL |
| `ghcr.io/llm-d/llm-d-rocm` | AMD ROCm |
| `ghcr.io/llm-d/llm-d-xpu` / `llm-d-hpu` / `llm-d-cpu` | Intel XPU / Gaudi / CPU |

These are described as "vLLM images with features not yet merged upstream." For non-vLLM engines, llm-d points users at the **upstream** images:

- **SGLang → `lmsysorg/sglang`** (e.g. `lmsysorg/sglang:v0.5.10.post1`) — upstream, *not* an llm-d image.
- **TensorRT-LLM → no llm-d path at all** today.

**Routing/control-plane images llm-d does publish (engine-neutral):** `llm-d-inference-scheduler` (EPP), `llm-d-routing-sidecar`, `llm-d-uds-tokenizer`. KServe already consumes these in `config-llm-scheduler.yaml` / decode presets.

> [!NOTE]
> Because there is no native image, KServe presets for SGLang must reference `lmsysorg/sglang:<tag>`, and TensorRT-LLM presets must reference `nvcr.io/nvidia/tensorrt-llm/release` (NGC auth required) or a Triton image. This also means KServe inherits each engine's image cadence and CVE posture directly from upstream, with no llm-d hardening layer.

---

## 6. Per-Runtime Deep Dive

### 6.1 SGLang — *officially supported by llm-d (v0.6.0)*

- **Status in llm-d:** Delivered under EPIC [llm-d#403](https://github.com/llm-d/llm-d/issues/403). SGLang options exist across the inference-scheduling, P/D-disaggregation, optimized-baseline, tiered-prefix-cache, and precise-prefix-cache-routing "well-lit paths." A `llm-d-sglang-overview` Grafana dashboard ships in-repo. (Caveat: GPU-only.)
- **Launch:** `python -m sglang.launch_server --model-path /mnt/models --served-model-name <name> --host 0.0.0.0 --port 30000 [--tp-size N --dp-size N --pp-size N]`.
- **Health:** `/health` (liveness, lightweight) and `/health_generate` (readiness, runs a real forward pass). ⚠️ On some versions these behaved identically ([sglang#12731](https://github.com/sgl-project/sglang/issues/12731)) — probe config must be validated against the pinned version.
- **Metrics:** `--enable-metrics`, served at `/metrics`. Queue gauge is **`sglang:num_queue_reqs`**. ⚠️ **v0.5.4+ changed the prefix `sglang:` → `sglang_`** — scrape config and EPP metric names must track the running version.
- **Multi-node:** `--nnodes`, `--node-rank`, `--dist-init-addr`; `--tp-size` is the *total* TP across nodes. Maps cleanly to LeaderWorkerSet.
- **P/D:** native, `--disaggregation-mode prefill|decode` with `--disaggregation-transfer-backend mooncake|nixl`; unified by `sglang_router`. In llm-d, the routing sidecar uses **`--connector=sglang`**.
- **LoRA:** `--lora-paths`, `--max-loras-per-batch`, `--max-lora-rank`.

### 6.2 TensorRT-LLM — *recognized in protocol, not yet a tested llm-d path*

- **Status in llm-d:** **No recipe, no CI, no guide** — open enhancement [llm-d#1550](https://github.com/llm-d/llm-d/issues/1550). The GIE metric mapping (`trtllm-serve`, `triton-tensorrt-llm`) is defined, and design docs name TRT-LLM as a target engine, but nothing is wired end-to-end.
- **Modern path:** `trtllm-serve <model> --backend pytorch [--tp_size N --pp_size N --ep_size N --host 0.0.0.0 --port 8000]`. **With `--backend pytorch` it loads an HF checkpoint directly — no `trtllm-build` AOT compile step.** This removes the single biggest historical obstacle to treating TRT-LLM like vLLM. (The `--backend tensorrt` path still needs the AOT engine build.)
- **Health:** `trtllm-serve` exposes `/health`; Triton exposes `/v2/health/ready` + `/v2/health/live`.
- **Metrics — the weak spot:** bare `trtllm-serve` `/metrics` returns iteration stats only and has historically blocked under load; robust Prometheus comes from **Triton** (`:8002/metrics`, `nv_inference_*` / `nv_trt_llm_*`) or from running under **NVIDIA Dynamo** (`trtllm_*`). Native `llm_*` Prometheus metrics are tracked in [TensorRT-LLM#4926](https://github.com/NVIDIA/TensorRT-LLM/issues/4926).
- **Multi-node:** MPI/`mpirun`; Triton leader/worker via LeaderWorkerSet.
- **P/D:** `trtllm-serve disaggregated` with separate context/generation servers; KV transfer via NIXL/UCX/MPI. **No llm-d routing-sidecar connector exists**, so KServe's P/D presets cannot drive TRT-LLM P/D without new glue.
- **Ecosystem note — Dynamo:** NVIDIA pushes **Dynamo** as its K8s orchestration/disaggregation layer *above* the engines, and is **collaborating** (not purely competing) with llm-d — notably contributing **NIXL** and committing to integrate the Dynamo Planner/KV-Cache-Manager into llm-d. For KServe, TRT-LLM is the engine; Dynamo is the NVIDIA-preferred orchestration option; llm-d shares components with it.

---

## 7. Feature Support Matrix (Target State in KServe)

| Feature | vLLM (today) | SGLang (achievable) | TensorRT-LLM (achievable) |
|---|---|---|---|
| Single-node serve | ✅ shipped | ✅ new preset | ✅ new preset (`--backend pytorch`) |
| Multi-node (LWS) | ✅ | ✅ new worker preset | ⚠️ MPI/Triton — harder, Triton or Dynamo |
| OpenAI routing via EPP | ✅ | ✅ engine-type=sglang | ✅ engine-type=trtllm-serve |
| Load/queue/KV-util routing (metrics) | ✅ | ✅ metric-name config | ✅ via Triton/Dynamo metrics |
| Precise prefix-cache routing (ZMQ KV events) | ✅ | ✅ | ❌ not wired |
| LoRA-affinity routing | ✅ | ⚠️ partial (no affinity scorer) | ❌ |
| P/D disaggregation | ✅ NIXL | ✅ `--connector=sglang` | ❌ no llm-d connector (Dynamo path) |
| Autoscale / scale-to-zero (HPA/KEDA) | ✅ | ✅ (metric-name change) | ✅ (metric source change) |
| Native llm-d hardened image | ✅ | ❌ upstream `lmsysorg/sglang` | ❌ `nvcr.io` |

---

## 8. Gaps, Risks & Recommendation

**Low-risk / high-value (do first):**
- SGLang single-node + multi-node presets and engine-type metric wiring. This is "follow the vLLM pattern with different flags," and llm-d has already proven the path.
- TensorRT-LLM single-node preset via `trtllm-serve --backend pytorch`, with EPP metrics sourced appropriately.

**Medium-risk:**
- Port hardcoding (`workload.go:8000`) and `/health` probe assumptions — need a per-engine indirection so SGLang's 30000 / `/health_generate` and Triton's `/v2/health/ready` work without the user hand-writing every field.
- Metric-name drift (SGLang `sglang:` → `sglang_`; TRT-LLM metrics maturing) — pin versions and document.

**High-risk / defer:**
- TensorRT-LLM P/D disaggregation and prefix-cache routing — no llm-d connector / no KV-event publishing wired. Likely a Dynamo-mediated path, out of scope for the first milestone.
- LoRA-affinity routing for non-vLLM engines — upstream-blocked.

**Recommendation:** Pursue a **phased adoption** — (1) SGLang single + multi-node + smart routing, (2) TensorRT-LLM single-node + smart routing via Triton/Dynamo metrics, (3) advanced P/D and prefix-cache as upstream support matures. The engineering mechanics, file-by-file, are detailed in the companion document: [multi_runtime_support_required_changes.md](./multi_runtime_support_required_changes.md).

---

## 9. References

- KServe LLMISVC types — `pkg/apis/serving/v1alpha2/llm_inference_service_types.go`
- Preset configs — `config/llmisvcconfig/config-llm-*.yaml`
- Workload reconcilers — `pkg/controller/v1alpha2/llmisvc/workload*.go`
- Scheduler/EPP reconciler — `pkg/controller/v1alpha2/llmisvc/scheduler.go`, `router.go`
- GIE Model Server Protocol — https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/docs/proposals/003-model-server-protocol/README.md
- llm-d artifacts/images — https://github.com/llm-d/llm-d/blob/main/docs/getting-started/artifacts.md
- llm-d model-server protocol + engine-type label — https://github.com/llm-d/llm-d/blob/main/docs/architecture/core/model-servers.md
- llm-d SGLang EPIC — https://github.com/llm-d/llm-d/issues/403
- llm-d TensorRT-LLM request — https://github.com/llm-d/llm-d/issues/1550
- SGLang server args — https://docs.sglang.io/advanced_features/server_arguments.html
- SGLang production metrics — https://github.com/sgl-project/sglang/blob/main/docs/references/production_metrics.md
- SGLang P/D disaggregation — https://docs.sglang.io/advanced_features/pd_disaggregation.html
- trtllm-serve — https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html
- TensorRT-LLM Prometheus metrics issue — https://github.com/NVIDIA/TensorRT-LLM/issues/4926
- NVIDIA Dynamo ↔ llm-d — https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference/
