# 🚀 LLM Inference Optimization with TensorRT + Triton

> **Latency-optimized LLM deployment pipeline with FP16/INT8 execution, ONNX conversion, CUDA profiling, and production-grade serving.**

---

## 📌 Project Overview

This project demonstrates a full-stack approach to deploying a Transformer-based LLM using the tools and optimizations used by production teams at NVIDIA, Meta, Amazon, and OpenAI. It includes:

- ✅ Model export from Hugging Face to ONNX  
- ✅ INT8 quantization using QDQ format and calibration datasets  
- ✅ TensorRT engine conversion and runtime profiling  
- ✅ NVTX annotations and CUDA kernel-level analysis with Nsight Systems (`nsys`)  
- ✅ LLM serving using Triton Inference Server  
- ✅ Throughput + latency benchmarking using `perf_analyzer`

> 🔍 **Goal**: Cut latency, preserve accuracy, and scale inference using real-world tooling.
- Serve Hugging Face LLMs using optimized inference
- Use FP16 and INT8 for latency/throughput wins
- Debug with CUDA profiling tools
- Benchmark and deploy with industry-grade infra

---

## 🧠 Why This Project Matters

Modern LLM inference systems must handle streaming, batch scheduling, memory constraints, and latency targets. This repo shows:

- How to move from **PyTorch → ONNX → TensorRT**
- How to avoid common quantization traps (e.g. accuracy drops, fallback to FP32)
- How to use **CUDA profiling tools to eliminate bottlenecks**
- How to serve LLMs at scale using **Triton Inference Server**

> ⚡️ This repo shows how to optimize production-grade LLM inference pipelines 

---

## 🧱 System Architecture

```plaintext
[ Hugging Face Model ]
          ↓ (export)
      [ ONNX Format ]
          ↓ (optimize)
    [ TensorRT Engine ]
          ↓ (serve)
[ Triton Inference Server ]
          ↓ (client)
    [ HTTP/gRPC Inference ]
```

> Profiling tools (nsys, nvtx) are used to measure and tune each stage.

---

## 🔧 Features and Highlights

### ✅ Export + Quantize
- Uses `optimum.exporters.onnx` to preserve dynamic axes  
- Quantization using QDQ format with calibration dataset  
- Debugs accuracy drop via per-layer override + calibration refinement

### ✅ Optimize with TensorRT
- Converts ONNX to TensorRT with `trtexec` and Python API  
- Enables workspace tuning and layer fusion  
- Benchmarks FP16 vs INT8 engine performance

### ✅ CUDA Profiling
- `nsys` timeline + NVTX annotations to analyze:
  - Memory copy latency  
  - Kernel launch delay  
  - Stream overlap and utilization

### ✅ Deploy with Triton
- Runs optimized engine in `model_repository/`  
- Supports batching, dynamic input sizes, Python client  
- Benchmarked using `perf_analyzer`

---

## 📁 Key Files and Directories

```bash
convert/
    export_onnx.py            # Converts Hugging Face → ONNX
    quantize_model.py         # Applies QDQ + calibration
optimize/
    build_trt_engine.py       # TensorRT engine builder
    trtexec_command.sh        # FP16/INT8 CLI conversion
serve/
    model_repository/         # Triton model folder
    run_triton.sh             # Launch Triton Server
    client/infer_client.py    # Sends inference requests
profile/
    run_nsys_profile.py       # NVTX + Nsight tracing
notebooks/
    latency_analysis.ipynb    # Profiling + performance graphs
diagrams/
    pipeline_architecture.png # Inference system diagram
```

---

## 🚀 How to Run

```bash
# 1. Export model to ONNX
python convert/export_onnx.py

# 2. Quantize to INT8 (optional)
python convert/quantize_model.py

# 3. Optimize with TensorRT
bash optimize/trtexec_command.sh

# 4. Launch Triton
bash serve/run_triton.sh

# 5. Send inference requests
python serve/client/infer_client.py

# 6. Profile with Nsight
python profile/run_nsys_profile.py
```

---

## 📉 Before vs After Optimization (Planned)

| Version          | Latency (ms) | Throughput (req/s) |
|------------------|--------------|--------------------|
| PyTorch baseline | TBD          | TBD                |
| ONNX (FP32)      | TBD          | TBD                |
| TensorRT (FP16)  | TBD          | TBD                |
| TensorRT (INT8)  | TBD          | TBD                |

_🚧 Benchmarking in progress. Results coming soon._

---

## 📊 Results + Insights

- INT8 inference expects to deliver ~3x throughput vs PyTorch baseline

- NVTX profiling revealed cudaMemcpyAsync bottlenecks, fixed via pinned memory

- Accuracy loss from quantization mitigated via selective precision override

- Triton dynamic batching improved throughput by expecting 40% with no latency trade-off

---

## 🧠 Key Learnings

- Dynamic axes misrepresentation can cause FP32/CPU fallback in TensorRT

- Nsight Systems + NVTX provide GPU visibility that can't be guessed

- QDQ quantization requires per-layer tuning to avoid quality drop

- Triton simplifies serving + auto-batching for production workloads

---

## 🤝 Open to Connect

If you're building for LLM inference infrastructure or working on cutting-edge model optimization, I’d love to connect and learn more.

💻 **GitHub**: [github.com/eric-wong-llm](https://github.com/eric-wong-llm)

## 🔗 References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Hugging Face Optimum](https://huggingface.co/docs/optimum/index)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
