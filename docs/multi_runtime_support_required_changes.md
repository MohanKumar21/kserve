# KServe: Engineering Changes Required to Support SGLang & TensorRT-LLM Runtimes

* **Author:** Mohan Kumar
* **Status:** Proposed
* **Target Version:** v0.17+
* **Companion Doc:** [multi_runtime_support_analysis.md](./multi_runtime_support_analysis.md)

---

## 1. Strategy: Preset-First, Minimal Core Change

The `LLMInferenceService` controller already delegates the entire `PodSpec` to the user/preset (`d.Spec.Template.Spec = *llmSvc.Spec.Template.DeepCopy()`). Because of this, the **majority of multi-runtime support ships as data (new `LLMInferenceServiceConfig` presets), not Go code.** The Go changes are limited to removing/parameterizing a handful of vLLM-specific assumptions.

We organize the work into three tiers:

| Tier | What | Code vs Config |
|---|---|---|
| **A. Presets** | New `LLMInferenceServiceConfig` objects for SGLang & TRT-LLM (single-node, multi-node, scheduler/metrics) | Config only |
| **B. Controller plumbing** | Engine-type label, port indirection, probe indirection, EPP metric defaults | Small Go changes |
| **C. Advanced** | P/D connectors, prefix-cache KV events, LoRA affinity for non-vLLM | Larger / upstream-dependent |

> [!NOTE]
> A user can run SGLang/TRT-LLM **today** by hand-writing a full `spec.template`. Everything below is about making it a **supported, ergonomic, routing-integrated** path rather than a DIY override.

---

## 2. Change Set Overview

| # | Area | File(s) | Type | Tier |
|---|---|---|---|---|
| 1 | SGLang single-node preset | `config/llmisvcconfig/config-llm-sglang-template.yaml` (new) | Config | A |
| 2 | SGLang multi-node worker preset | `config/llmisvcconfig/config-llm-sglang-worker.yaml` (new) | Config | A |
| 3 | TRT-LLM single-node preset | `config/llmisvcconfig/config-llm-trtllm-template.yaml` (new) | Config | A |
| 4 | Engine-type label propagation | `scheduler.go`, `workload*.go`, types | Go | B |
| 5 | EPP per-engine metric config (sglang/trtllm) | `scheduler.go` (`withCoreMetricsExtractorPlugin`) | Go | B |
| 6 | Workload Service port indirection (un-hardcode 8000) | `workload.go` | Go | B |
| 7 | Probe path/port indirection | presets + (optional) defaults | Config/Go | B |
| 8 | Parallelism flag mapping per engine | presets (templating) | Config | A/B |
| 9 | Preset selection wiring | `config_merge.go` | Go | B |
| 10 | SGLang P/D connector preset | decode/prefill presets (new) | Config | C |
| 11 | Charts/manifests packaging | `charts/kserve-runtime-configs/...`, `config/` kustomize | Config | A |
| 12 | E2E + docs + samples | `test/e2e/llmisvc`, `docs/samples/llmisvc` | Test/Docs | A/B |

---

## 3. Tier A — New Runtime Presets

### 3.1 SGLang single-node preset (new)

`config/llmisvcconfig/config-llm-sglang-template.yaml`:

```yaml
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceServiceConfig
metadata:
  name: kserve-config-llm-sglang-template
spec:
  annotations:
    serving.kserve.io/model-based-routing-enabled: "true"
  template:
    containers:
      - name: main
        image: lmsysorg/sglang:v0.5.10.post1        # upstream — no native llm-d image (see analysis §5)
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8000                      # align to KServe's expected 8000 (override SGLang's 30000)
            protocol: TCP
        command: ["/bin/bash", "-c"]
        args:
          - |-
            exec python3 -m sglang.launch_server \
              --model-path /mnt/models \
              --served-model-name {{ .Spec.Model.Name }} \
              --host 0.0.0.0 \
              --port 8000 \
              --enable-metrics \
              {{- if .Spec.Parallelism }}
              {{- if .Spec.Parallelism.Tensor }}--tp-size {{ .Spec.Parallelism.Tensor }}{{ end }} \
              {{- if .Spec.Parallelism.Data }}--dp-size {{ .Spec.Parallelism.Data }}{{ end }} \
              {{- if .Spec.Parallelism.Pipeline }}--pp-size {{ .Spec.Parallelism.Pipeline }}{{ end }} \
              {{- end }}
              ${SGLANG_ADDITIONAL_ARGS:-}
        env:
          - name: HF_HUB_CACHE
            value: /models
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
          failureThreshold: 10
        readinessProbe:
          httpGet: { path: /health_generate, port: 8000 }   # real forward-pass readiness
          periodSeconds: 5
          failureThreshold: 3
        startupProbe:
          httpGet: { path: /health, port: 8000 }
          periodSeconds: 10
          failureThreshold: 60
```

> [!WARNING]
> **Two SGLang-specific gotchas to encode and document:**
> 1. **Port:** SGLang defaults to **30000**; we force `--port 8000` to match KServe's hardcoded Service port (see change #6). If we instead un-hardcode the port, this preset can use 30000.
> 2. **Metric prefix drift:** SGLang **v0.5.4+ renamed `sglang:` → `sglang_`**. The EPP metric config (change #5) and any dashboards must match the pinned image tag. Pin the image tag in the preset and the metric names together.
> 3. **`/health` vs `/health_generate`:** validate against the pinned version ([sglang#12731](https://github.com/sgl-project/sglang/issues/12731)) — on some builds `/health` also runs generation and can flap under load.

### 3.2 SGLang multi-node worker preset (new)

`config/llmisvcconfig/config-llm-sglang-worker.yaml` — mirrors `config-llm-worker-data-parallel.yaml` but with SGLang's distributed flags:

```yaml
# leader (spec.template) and worker (spec.worker) share dist-init-addr from LWS_LEADER_ADDRESS
args:
  - |-
    exec python3 -m sglang.launch_server \
      --model-path /mnt/models \
      --tp-size {{ .Spec.Parallelism.Tensor }} \
      --nnodes {{ .Spec.Parallelism.DataLocal }} \
      --node-rank ${LWS_WORKER_INDEX:-0} \
      --dist-init-addr ${LWS_LEADER_ADDRESS}:50000 \
      --host 0.0.0.0 --port 8000 --enable-metrics
```

### 3.3 TensorRT-LLM single-node preset (new)

`config/llmisvcconfig/config-llm-trtllm-template.yaml`:

```yaml
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceServiceConfig
metadata:
  name: kserve-config-llm-trtllm-template
spec:
  template:
    containers:
      - name: main
        image: nvcr.io/nvidia/tensorrt-llm/release    # NGC auth required (imagePullSecret)
        ports:
          - containerPort: 8000
        command: ["/bin/bash", "-c"]
        args:
          - |-
            exec trtllm-serve /mnt/models \
              --backend pytorch \
              --host 0.0.0.0 --port 8000 \
              {{- if .Spec.Parallelism.Tensor }}--tp_size {{ .Spec.Parallelism.Tensor }}{{ end }} \
              {{- if .Spec.Parallelism.Pipeline }}--pp_size {{ .Spec.Parallelism.Pipeline }}{{ end }} \
              ${TRTLLM_ADDITIONAL_ARGS:-}
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
        readinessProbe:
          httpGet: { path: /health, port: 8000 }
```

> [!IMPORTANT]
> - `--backend pytorch` loads an **HF checkpoint directly — no `trtllm-build` step**. This is what makes a vLLM-style preset viable. The `tensorrt` backend would require an extra engine-build init-container and is out of scope for v1.
> - **NGC image pull:** TRT-LLM images live on `nvcr.io` and need an `imagePullSecret`; document this prominently.
> - **Metrics caveat:** bare `trtllm-serve` `/metrics` is weak. For real routing signals either run a **Triton** variant (metrics on `:8002`) or run under **Dynamo** (`trtllm_*` metrics). The EPP metric config (change #5) must point at whichever source is in use.

---

## 4. Tier B — Controller Plumbing

### 4.1 Engine-type label propagation (change #4)

The EPP already reads `inference.networking.k8s.io/engine-type` and defaults to `vllm` ([`scheduler.go:1510-1529`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/scheduler.go)). We need the controller to **stamp this label on workload pods** based on the runtime.

**Proposed:** add an optional discriminator to the API and propagate it as a pod label + EPP `defaultEngine`:

```go
// pkg/apis/serving/v1alpha2/llm_inference_service_types.go
type LLMInferenceServiceSpec struct {
    // +optional
    // Runtime selects the inference engine for metric/label wiring.
    // One of: vllm (default), sglang, trtllm-serve, triton-tensorrt-llm.
    Runtime string `json:"runtime,omitempty"`
    // ...
}
```

- Workload reconcilers (`workload_single_node.go`, `workload_multi_node.go`) add `inference.networking.k8s.io/engine-type: <runtime>` to the pod template labels.
- `scheduler.go` sets `"defaultEngine": <runtime>` instead of the hardcoded `"vllm"`.
- Default empty → `vllm`, preserving current behavior (no breaking change).

### 4.2 EPP per-engine metric config (change #5)

`withCoreMetricsExtractorPlugin` (`scheduler.go:1483`) currently emits a single `engineConfig` named `"vllm"`. Extend it with a built-in table of the GIE metric mappings so each runtime gets correct metric names without the user hand-writing them:

```go
var engineMetricDefaults = map[string]map[string]string{
    "vllm": {
        "queuedRequestsSpec":  "vllm:num_requests_waiting",
        "runningRequestsSpec": "vllm:num_requests_running",
        "kvUsageSpec":         "vllm:kv_cache_usage_perc",
    },
    "sglang": {
        "queuedRequestsSpec":  "sglang:num_queue_reqs",   // NOTE: sglang_ on v0.5.4+
        "runningRequestsSpec": "sglang:num_running_reqs",
        "kvUsageSpec":         "sglang:token_usage",
    },
    "trtllm-serve": {
        "queuedRequestsSpec":  "trtllm_num_requests_waiting",
        "runningRequestsSpec": "trtllm_num_requests_running",
        "kvUsageSpec":         "trtllm_kv_cache_utilization",
    },
}
```

The user can still override via the existing inline `EndpointPickerConfig` mechanism; this just provides correct defaults. **No EPP/GIE image change is needed** — the GIE protocol is engine-agnostic (see analysis §4).

### 4.3 Un-hardcode the workload Service port (change #6)

[`workload.go:131-148`](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/workload.go) pins the Service to `8000` with an explicit comment that changing it touches "presets, routing sidecar flags, routing sidecar container spec." Two options:

- **Option A (minimal, recommended for v1):** keep `8000` and require runtime presets to bind their server to `8000` (as the SGLang preset above does with `--port 8000`). Zero core change; documented constraint.
- **Option B (clean, later):** derive the port from the main container's first `containerPort` (or a new `spec.router.targetPort`) and thread it through the Service, probes, InferencePool `targetPorts`, and routing-sidecar flags.

Recommend shipping **Option A** first, tracking **Option B** as a follow-up, since Option B touches the P/D routing sidecar contract.

### 4.4 Probe indirection (change #7)

Health-path differences (`/health` vs `/health_generate` vs `/v2/health/ready`) are fully expressible in the preset's `PodSpec` (as shown in §3), so **no core change is strictly required** — the presets carry the right probes. Only if we add server-side probe defaulting would Go change be needed; not recommended for v1.

### 4.5 Preset selection wiring (change #9)

`config_merge.go` auto-selects the vLLM presets by deployment shape. Extend the selection so that when `spec.runtime` (change #4.1) is `sglang`/`trtllm-serve`, the controller merges the corresponding `kserve-config-llm-<runtime>-*` preset instead of the vLLM one. Keep `WellKnownDefaultConfigs` additive and backward-compatible.

---

## 5. Parallelism & LoRA Flag Mapping (reference for preset authors)

| Concept | vLLM | SGLang | TensorRT-LLM (`trtllm-serve`) |
|---|---|---|---|
| Tensor parallel | `--tensor-parallel-size` | `--tp-size` | `--tp_size` |
| Pipeline parallel | `--pipeline-parallel-size` | `--pp-size` | `--pp_size` |
| Data parallel | `--data-parallel-size` | `--dp-size` | (n/a; use ep/replicas) |
| Expert parallel | `--enable-expert-parallel` | `--ep-size` | `--ep_size` |
| Multi-node addr | `--data-parallel-address` | `--dist-init-addr` | MPI/`mpirun` |
| Node count / rank | (LWS env) | `--nnodes` / `--node-rank` | MPI hostfile |
| Served name | `--served-model-name` | `--served-model-name` | (model id) |
| Max LoRA rank | `--max-lora-rank` | `--max-lora-rank` | (engine-build) |
| LoRA adapters | `--max-loras` / `--max-cpu-loras` | `--max-loras-per-batch` / `--lora-paths` | limited |
| Port | `--port` | `--port` (def 30000) | `--port` (def 8000) |
| Metrics enable | (always on) | `--enable-metrics` | (Triton/Dynamo) |

The `spec.parallelism.*` fields ([`llm_inference_service_types.go`](file:///Users/mohankumar/kluisz/kserve/pkg/apis/serving/v1alpha2/llm_inference_service_types.go)) already capture the *intent*; only the **flag spelling in the preset template** differs per engine. No API change needed for parallelism.

---

## 6. Tier C — Advanced Features (Phase 2+)

| Feature | SGLang | TensorRT-LLM | Required work |
|---|---|---|---|
| **P/D disaggregation** | `--disaggregation-mode`, routing sidecar `--connector=sglang` | `trtllm-serve disaggregated` (NIXL/UCX) | New decode/prefill presets; for TRT-LLM no llm-d connector exists → likely Dynamo-mediated, defer |
| **Precise prefix-cache routing** | ZMQ KV events supported | not wired | SGLang: add `precise-prefix-cache-scorer` config (mirror `e2e-gpt-oss/llmisvc_config_prefix_cache.yaml`); TRT-LLM: blocked |
| **LoRA-affinity scoring** | partial | ❌ | Upstream GIE LoRA scorer is vLLM-biased; defer |
| **Multi-node TRT-LLM** | n/a | MPI/Triton leader-worker (LWS) | Separate Triton-based preset + MPI bootstrap; higher effort |

---

## 7. Packaging, Testing & Docs (change #11, #12)

- **Charts/kustomize:** add the new presets to `charts/kserve-runtime-configs/files/llmisvcconfigs/` and `config/llmisvcconfig/kustomization.yaml` so they install alongside the vLLM presets.
- **Samples:** add `docs/samples/llmisvc/sglang-*` and `docs/samples/llmisvc/trtllm-*` mirroring the existing `e2e-gpt-oss/` and `qwen-llm-d-envoy-ai-gateway/` examples (including the Envoy AI Gateway `AIGatewayRoute` wiring, which is engine-agnostic and needs no change).
- **E2E:** extend `test/e2e/llmisvc` with an SGLang single-node smoke test first (cheapest, llm-d-proven), then TRT-LLM.
- **Webhook validation:** if `spec.runtime` is added, validate the enum in `pkg/webhook/admission/llminferenceservice`.

---

## 8. Phased Roadmap & Effort

| Phase | Scope | Effort | Risk |
|---|---|---|---|
| **1** | SGLang single-node preset + engine-type label + EPP metric defaults + sample + e2e | ~1 preset, ~3 small Go diffs | Low (llm-d-proven) |
| **2** | SGLang multi-node (LWS) preset + smart routing validation | 1 preset | Low–Med |
| **3** | TensorRT-LLM single-node (`trtllm-serve --backend pytorch`) + Triton/Dynamo metric wiring + NGC pull docs | 1 preset + metric config | Med |
| **4** | SGLang P/D disaggregation (`--connector=sglang`) + precise prefix-cache routing | 2 presets + EPP config | Med |
| **5** | Port indirection (Option B), TRT-LLM multi-node & P/D (Dynamo), LoRA affinity | Core change + upstream deps | High |

> [!NOTE]
> Phases 1–3 deliver "serve any of the three major runtimes with smart (queue/load/KV-util) routing" — the headline goal — with **only additive config and three small, backward-compatible Go changes**. Phases 4–5 chase the advanced disaggregation/caching features and are gated on upstream llm-d/Dynamo maturity.

---

## 9. Summary of Required Code Changes (the short list)

1. **`spec.runtime` enum field** + defaulting to `vllm` — `llm_inference_service_types.go`, defaults, webhook.
2. **Stamp `inference.networking.k8s.io/engine-type` pod label** from `spec.runtime` — `workload_single_node.go`, `workload_multi_node.go`.
3. **`defaultEngine` + per-engine metric defaults** in EPP config — `scheduler.go` (`withCoreMetricsExtractorPlugin`, `engineMetricDefaults` table).
4. **Preset selection by runtime** — `config_merge.go`.
5. *(Optional, Phase 5)* **Port indirection** — `workload.go` + routing-sidecar flags.

Everything else is **net-new YAML presets, samples, charts, e2e tests, and docs** — no rearchitecting of the controller.
