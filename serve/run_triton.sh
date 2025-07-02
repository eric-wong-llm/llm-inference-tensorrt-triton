#!/bin/bash

# Explicitly pull the Triton image
docker pull nvcr.io/nvidia/tritonserver:23.05-py3

# Run the Triton Inference Server container
docker run --rm --gpus all \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.05-py3 \
  tritonserver --model-repository=/models
