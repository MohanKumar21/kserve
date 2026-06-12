# KServe Proposal: Supporting NVIDIA Dynamic Resource Allocation (DRA) in LLMInferenceService

* **Author:** Mohan Kumar
* **Status:** Proposed
* **Target Version:** v0.17+

---

## 1. Overview & Problem Statement

Traditionally, Kubernetes clusters allocate GPU accelerators to workloads using the **Device Plugin Framework** (e.g., requesting resources via `nvidia.com/gpu` under `resources.limits`). While functional, the device plugin approach imposes several limitations:
- **Static and Binary Allocation**: A GPU is typically assigned entirely to a container. Fine-grained slicing, Multi-Process Service (MPS) sharing, or Multi-Instance GPU (MIG) allocations require static configuration of kubelet settings or host daemonsets.
- **Lack of Topology Co-Scheduling**: High-performance LLM training and inference depend heavily on host interconnect topologies (NVLink, NVSwitch) and network interfaces (RDMA/RoCE). Static device plugin allocation does not align GPU scheduler allocations with network topologies.
- **Inflexible Parameterization**: Wavelength or VRAM parameters cannot be dynamically requested per workload.

**Dynamic Resource Allocation (DRA)** (introduced as beta in Kubernetes 1.30/1.31) addresses these limitations. DRA uses a resource-class driver model (`gpu.nvidia.com`), allowing workloads to request GPUs declaratively using `ResourceClaims` and `ResourceClaimTemplates`. This enables:
1. **Dynamic Sharing and Slicing**: Pods can request fractional GPUs or specific MIG profiles directly inside their manifests.
2. **Topology-Aware Scheduling**: Integrates with cluster schedulers (like Kueue and scheduling plugins) to co-schedule network and GPU resources.

This proposal documents and formalizes support for NVIDIA DRA within KServe's `LLMInferenceService`.

---

## 2. Controller Architecture & Structural Compatibility

KServe's `LLMInferenceService` (under the `v1alpha2` API version) is uniquely suited to adopt Kubernetes DRA because of its template delegation design.

### How KServe Reconciles Workloads
When a user defines an `LLMInferenceService`, the reconciler creates Kubernetes `Deployment` resources (for single-node serving) or `LeaderWorkerSet` resources (for multi-node serving). The Pod template specs are generated using:
- `spec.template` for single-node main/decode workloads or multi-node leaders.
- `spec.worker` for multi-node workers.
- `spec.prefill` for disaggregated prefill workloads.

Inside the controller reconcilers (e.g., [workload_single_node.go](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/workload_single_node.go) and [workload_multi_node.go](file:///Users/mohankumar/kluisz/kserve/pkg/controller/v1alpha2/llmisvc/workload_multi_node.go)), the controller performs a copy of the user's `PodSpec` configuration:
```go
d.Spec.Template.Spec = *llmSvc.Spec.Template.DeepCopy()
```
Since the Kubernetes API Go struct `corev1.PodSpec` natively supports `ResourceClaims` and `corev1.ResourceRequirements` supports `Claims`, **the KServe controller propagates DRA claims to the underlying resources automatically without requiring any codebase modifications.**

---

## 3. Migration Guide: From Device Plugins to DRA

Migrating an existing `LLMInferenceService` from classic device plugins (`nvidia.com/gpu`) to DRA requires a shift in how resource requirements are declared.

### Legacy Configuration (Device Plugin)
In legacy setups, GPUs were requested under `limits` and `requests`:
```yaml
spec:
  template:
    spec:
      containers:
      - name: main
        resources:
          limits:
            nvidia.com/gpu: "1"
          requests:
            nvidia.com/gpu: "1"
```

### DRA Configuration
In DRA setups, the GPU is declared as a pod-level `ResourceClaim` (via a template) and then bound to the container under `resources.claims`:
```yaml
spec:
  template:
    spec:
      resourceClaims:
      - name: gpu-claim
        resourceClaimTemplateName: qwen-gpu-template
      containers:
      - name: main
        resources:
          claims:
          - name: gpu-claim
```

---

## 4. Key Design Constraints & Deep Dive

### A. Replica Scaling & Mandatory Use of `ResourceClaimTemplate`
Kubernetes DRA supports two ways to specify claims:
1. `resourceClaimName`: Reference a pre-created, cluster-scoped `ResourceClaim`.
2. `resourceClaimTemplateName`: Reference a `ResourceClaimTemplate` that dynamically provisions a unique `ResourceClaim` per pod.

> [!WARNING]
> Because KServe's `LLMInferenceService` supports dynamic scaling (either static `replicas` > 1 or auto-scaling via KEDA/HPA), **users must never use `resourceClaimName` in the template**. Doing so causes all pod replicas to share a single claim name. The first pod will bind the resource, and all subsequent pods will fail scheduling and remain stuck in `Pending`. Always use `resourceClaimTemplateName`.

### B. Sidecar Container and InitContainer Allocations
One of the major benefits of DRA in KServe is the separation of container-level resource claims:
- **Main Container**: Needs a GPU for model inference. It requests the claim under its `resources.claims`.
- **KServe Routing Sidecar**: Injected by the KServe controller to handle model routing and health checks. It does not need a GPU and will not map the claim.
- **Storage Initializer**: Runs as an initContainer to pull model weights from remote storage (S3, HF, etc.). It does not need a GPU.

Under the device plugin system, a pod-level limit of `nvidia.com/gpu` could occasionally lead to device allocation confusion or unintended scheduling constraints. With DRA, the GPU is explicitly mapped only to the `main` container, preventing sidecars from claiming or locking GPU devices.

### C. Multi-Node (LeaderWorkerSet) Configurations
For multi-node LLM deployments (e.g., using `spec.worker` to create a `LeaderWorkerSet`):
- Both the `spec.template` (leader) and `spec.worker` (workers) must declare resource claims using `resourceClaimTemplateName`.
- Highly parallel configurations (TP/PP) rely on low-latency interconnects (NVLink). The cluster must run a topology-aware DRA scheduler (e.g., Kueue) to ensure the leader and worker pods are co-scheduled on nodes with physical GPU-to-GPU bridges.

### D. Autoscaling to Zero (KEDA)
When KServe scales an `LLMInferenceService` down to zero replicas:
- No pods exist, which means no `ResourceClaim` objects are provisioned by the `ResourceClaimTemplate`.
- When requests arrive and KEDA scales the workload up, `ResourceClaim` resources are dynamically provisioned by the template, allocating the GPU devices on-demand. This facilitates clean scale-to-zero behavior and prevents cluster-level GPU lockups.

---

## 5. Reference Manifests

### 1. ResourceClaimTemplate Definition
```yaml
apiVersion: resource.k8s.io/v1alpha3
kind: ResourceClaimTemplate
metadata:
  name: nvidia-gpu-template
  namespace: kserve-test
spec:
  spec:
    resourceClassName: gpu.nvidia.com
    devices:
      requests:
      - name: gpu
        deviceClassName: gpu.nvidia.com
        count: 1
```

### 2. LLMInferenceService using DRA
```yaml
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: qwen2-7b-instruct-dra
  namespace: kserve-test
spec:
  model:
    uri: hf://Qwen/Qwen2.5-7B-Instruct
    name: Qwen/Qwen2.5-7B-Instruct
  replicas: 1
  router:
    scheduler: {}
    route: {}
    gateway: {}
  template:
    spec:
      resourceClaims:
      - name: gpu
        resourceClaimTemplateName: nvidia-gpu-template
      containers:
      - name: main
        resources:
          requests:
            cpu: "2"
            memory: 16Gi
          limits:
            cpu: "4"
            memory: 32Gi
          claims:
          - name: gpu
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
            scheme: HTTPS
          initialDelaySeconds: 120
          periodSeconds: 30
          timeoutSeconds: 30
          failureThreshold: 5
```
