# 🚀 LLM Inference Optimization with TensorRT + Triton

> **End-to-end deployment of Transformer-based LLMs with INT8 quantization, CUDA-level profiling, and production-grade Triton serving pipeline.**

---

## 📌 Overview

This project showcases a production-oriented LLM inference pipeline, integrating key optimizations used by real-world teams at NVIDIA, Meta, Amazon, and OpenAI. It includes:

- ✅ Hugging Face → ONNX model export with dynamic axis support
- ✅ INT8 quantization via QDQ format and calibration dataset
- ✅ TensorRT engine build + performance benchmarking
- ✅ Kernel-level analysis with Nsight Systems (nsys) and NVTX annotations
- ✅ Scalable inference serving with Triton Inference Server
- ✅ Real throughput and latency benchmarking using perf_analyzer

> 🎯 **Objective**: Maximize inference speed while preserving accuracy and enabling deployment scalability.

---

## 💡 Why This Project Matters

Modern LLM inference faces growing challenges: high latency, memory bottlenecks, and limited scalability. This project addresses these by:

- Demonstrating a full pipeline: PyTorch → ONNX → TensorRT → Triton
- Diagnosing and mitigating quantization-induced accuracy drops
- Using Nsight Systems + NVTX for deep CUDA visibility
- Leveraging Triton's batching and concurrent stream support for deployment

> ⚡️ A hands-on, real-world implementation of the techniques powering today’s most efficient AI systems.

---

## 🔧 Key Features

### ✳️  ONNX Export + INT8 Quantization
- Model exported with dynamic shapes via optimum.exporters.onnx
- INT8 quantization using QDQ + calibration dataset
- Layer-level override to mitigate quality degradation

### ✳️  TensorRT Engine Optimization
- Uses both trtexec and Python API for engine builds
- Compares FP32 / FP16 / INT8 performance
- Workspace tuning, layer fusion, and dynamic batch support

### ✳️  CUDA Profiling & Bottleneck Detection
- Deep dive via nsys timeline + NVTX annotations
- Identifies:
  - cudaMemcpyAsync latency
  - Kernel launch delays
  - Underutilized GPU streams

### ✳️  Triton Inference Server Deployment
- Configured model_repository for dynamic input, batching
- Python client for benchmarking
- Production-style API interface (HTTP/gRPC)

---

## 🧱 Architecture Diagram

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

> Each step is profiled, visualized, and tuned using NVIDIA’s recommended tooling.

---

## 📁 Project Structure

```bash
convert/
    export_onnx.py            # Hugging Face → ONNX
    quantize_model.py         # QDQ quantization
optimize/
    build_trt_engine.py       # FP16/INT8 engine conversion
    trtexec_command.sh
serve/
    model_repository/         # Triton model setup
    run_triton.sh             # Launch Triton Server
    client/infer_client.py    # Python client for inference
profile/
    run_nsys_profile.py       # NVTX + Nsight Systems profiling
notebooks/
    latency_analysis.ipynb    # Performance plots
diagrams/
    pipeline_architecture.png # Inference system diagram

Note: `.onnx` and `.plan` model files are excluded from the repo due to size. Please run `convert/export_onnx.py` and `optimize/trtexec_command.sh` to regenerate them.
```

---

## 🚀 Quickstart

```bash
# 1. Export model to ONNX
python convert/export_onnx.py

# 2. Apply INT8 Quantization (optional)
python convert/quantize_model.py

# 3. Build Engine with TensorRT
bash optimize/trtexec_command.sh

# 4. Launch Triton Server
bash serve/run_triton.sh

# 5. Run Inference via Client
python serve/client/infer_client.py

# 6. Run Nsight Profiling
python profile/run_nsys_profile.py
```

---

## 📉 Results (In Progress)

| Engine Type      | Latency (ms) | Throughput (req/s) |
|------------------|--------------|--------------------|
| PyTorch Baseline | TBD          | TBD                |
| ONNX (FP32)      | TBD          | TBD                |
| TensorRT (FP16)  | TBD          | TBD                |
| TensorRT (INT8)  | TBD          | TBD                |

> Final benchmarks will include token throughput, memory footprint, and profiling overlays.

---

## 🔍 Insights & Debug Notes

- NVTX tracing revealed kernel-streaming bottlenecks in memory ops

- INT8 quality loss was mitigated by per-layer selective dequantization

- Triton's dynamic batching is expected to improve throughput by ~40% without latency cost

- PyTorch to ONNX export required careful handling of dynamic axes

---

## 🧠 Key Takeaways

- Understanding where latency comes from (e.g., memcpy, shape inference, fallback to FP32) is key to optimizing inference.

- CUDA profiling with Nsight is non-optional when deploying real-time LLMs.

- Triton makes scalable, production-grade LLM inference practical and maintainable.

---

## 🤝 Open to Connect

If you're building high-performance LLM inference infra or interested in discussing deployment tools, let's connect.

💻 **GitHub**: [github.com/eric-wong-llm](https://github.com/eric-wong-llm)
🔗 Open to feedback and collaboration!

---

## 🔗 References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Hugging Face Optimum](https://huggingface.co/docs/optimum/index)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
