# 🖥️ vast.ai GPU Sizing & Operational Guide (LLM Fine-tuning & vLLM)

> **Applies to:** Sprint 1 Task 1.12 (GPU Sizing Research), Sprint 4 (QLoRA SFT + DPO Alignment), and Sprint 5 (vLLM Serving Benchmark).

---

## 1. VRAM Sizing for 7B Vietnamese Cart-Recovery LLMs

Primary candidate base models: **`Qwen/Qwen2.5-7B-Instruct`** and **`Viet-Mistral/Vistral-7B-Chat`**.

| Workload | Method & Precision | Minimum VRAM | Recommended VRAM | Est. Wall-Clock Time (5K SFT / 2K DPO) |
|---|---|---|---|---|
| **SFT Fine-tuning** | QLoRA (4-bit NF4, `bfloat16`, Unsloth/PEFT, `seq_len=2048`) | ~14 GB | **24 GB** | ~35–50 mins on 1× RTX 4090 / A100 |
| **DPO Alignment** | QLoRA + Adapter-Off Reference (`TRL DPOTrainer`, `bfloat16`) | ~18 GB | **24–40 GB** | ~30–45 mins on 1× RTX 4090 / A100 |
| **LoRA Merge & AWQ Export** | FP16 merge + 4-bit AutoAWQ calibration | ~16 GB VRAM + 32 GB RAM | **24–48 GB** | ~15 mins |
| **vLLM Serving (Quantized)** | AWQ 4-bit (`gpu_memory_utilization=0.85`, `max_num_seqs=64`) | ~10–12 GB | **24 GB** | On-demand during benchmark / demo |
| **vLLM Serving (Full BF16)** | Unquantized `bfloat16` weights + PagedAttention KV cache | ~18–22 GB | **24–40 GB** | On-demand during benchmark / demo |

---

## 2. vast.ai GPU Comparison & Cost Breakdown

| GPU Model | VRAM | Memory Bandwidth | Avg. On-Demand Price | Avg. Interruptible (Spot) | Recommendation |
|---|---|---|---|---|---|
| **NVIDIA RTX 4090** | 24 GB | 1,008 GB/s | \$0.35 – \$0.55 / hr | \$0.22 – \$0.35 / hr | ⭐ **Best Cost/Performance** for QLoRA SFT, Unsloth DPO, and AWQ 4-bit vLLM serving (~$1–$2 per full training run). |
| **NVIDIA RTX A6000 / 6000 Ada** | 48 GB | 768 – 960 GB/s | \$0.50 – \$0.85 / hr | \$0.35 – \$0.55 / hr | ⭐ **Best Headroom** for larger DPO batch sizes, LoRA merging, and unquantized BF16 vLLM benchmarks. |
| **NVIDIA L40S** | 48 GB | 864 GB/s | \$0.75 – \$1.10 / hr | \$0.50 – \$0.75 / hr | Datacenter-grade inference GPU with native FP8 support for high-concurrency vLLM serving. |
| **NVIDIA A100 (40GB / 80GB)** | 40 / 80 GB | 1,555 – 2,039 GB/s | \$0.95 – \$1.60 / hr | \$0.65 – \$1.10 / hr | Ideal for full-parameter fine-tuning, though overkill compared to RTX 4090 / A6000 for 5K-sample QLoRA. |

---

## 3. Instance Selection Checklist on vast.ai

Always apply these filters in [cloud.vast.ai](https://cloud.vast.ai/) to avoid slow network pulls or unstable hosts:

- **Verified Machines Only:** `Checked`
- **Reliability Score:** `> 99.0%`
- **CUDA Version:** `>= 12.1` (Required by `vllm` and `flash-attn`)
- **Disk Space:** `>= 80 GB` SSD (Base 7B weights ~15 GB + checkpoints + container layers)
- **Internet Down/Up Speed:** `>= 500 Mbps` (Downloads 15 GB HuggingFace weights in <1 minute)
- **System RAM:** `>= 32 GB`

---

## 4. Step-by-Step Operational Runbook

### Step 1: Select Docker Image Template
```text
pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

### Step 2: Connect via SSH with Local Port Forwarding
```bash
# Forward remote vLLM port 8000 to local port 8000
ssh -p <VAST_SSH_PORT> root@<VAST_IP> -L 8000:localhost:8000
```

### Step 3: Upload Code & Compact LLM JSONL Datasets Only
Upload only lightweight `.jsonl` files (~10–20 MB) — **never** upload `data/raw/` (14 GB) to GPU instances:
```bash
scp -P <VAST_SSH_PORT> -r configs src data/processed/llm root@<VAST_IP>:/workspace/
```

### Step 4: Push Artifacts & Destroy Instance Immediately
- Configure `HF_TOKEN` so training scripts automatically push LoRA adapters / merged AWQ weights to a private HuggingFace Hub repository upon completion.
- **Cost Control Reminder:** Stopping an instance on vast.ai still incurs disk storage charges (\$/GB/month). Once checkpoints and metrics are pushed to HuggingFace Hub / MLflow, **Destroy** the instance to stop billing completely.
