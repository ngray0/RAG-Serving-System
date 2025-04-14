# RAG/ Serverless LLM Serving System with GPU Accelerated Information Retreival (IR)

This repository contains our implementation of a high-performance GPU-accelerated information retrieval system with a batch-oriented RAG (Retrieval-Augmented Generation) serving architecture. Our system achieves a maximum throughput of 23.67 requests/second on a single Nvidia Testla L4 GPU (batch size = 32) and offers support for scalable deployment/ autoscaling with Kubernetes.

## Directory Structure

```
RAG-Serving-System/
├── LICENSE
├── task-1/                 # GPU-accelerated Information Retrieval (IR) implementations
└── task-2/                 # Batched RAG Serving System
    ├── Dockerfile          # Container definition for the RAG service
    ├── Dockerfile.autoscaler  # Container definition for the autoscaler
    ├── main.py             # Main application entry point
    ├── requirements.txt    # Python dependencies
    ├── benchmarks/         # Load testing and performance evaluation tools
    │   ├── load_generator.py  # Async request generator for load testing
    │   ├── load_test.sh    # Script to run load tests with increasing RPS
    │   └── metrics/        # Performance metrics collection utilities
    │       └── collector.py  # Records and analyzes request latencies
    ├── data/               
    ├── deployment/         # Kubernetes deployment configurations
    │   ├── auto_scaler.py  # Horizontal pod autoscaler logic
    │   ├── autoscaler.yaml # K8s autoscaler definition
    │   ├── rag-service.yaml  # K8s RAG service deployment
    │   ├── rag-service-service.yaml  # K8s service definition
    │   └── redis.yaml      # Redis deployment for distributed queue
    ├── rag_service/        # Core RAG service implementation
    │   ├── config.py       # Configuration settings
    │   ├── api/            # FastAPI endpoints and models
    │   │   ├── endpoints.py  # API routes for RAG service
    │   │   └── models.py   # Pydantic models for request/response
    │   └── core/           # Core RAG functionality
    │       ├── batch_processor.py  # Processes batches of RAG requests
    │       ├── request_queue.py    # Request queue implementations
    │       └── retriever.py        # GPU-accelerated vector similarity search
    └── scripts/            # Utility scripts
        ├── fact_dataset.py    # Generates synthetic fact dataset
        └── squad_dataset.py   # Prepares SQuAD dataset with embeddings
```

## Key Features

- **GPU-Accelerated Vector Similarity**: Fast similarity computation using CuPy for efficient document retrieval
- **Batched Request Processing**: Optimised multi-stage pipeline for efficient GPU utilisation (batched embedding creation, document retrieval and LLM inference)
- **Asynchronous API**: Non-blocking request handling with Redis-based distributed queue
- **Horizontal Scaling**: Kubernetes-based autoscaling based on queue metrics
- **Comprehensive Benchmarking**: Load testing framework with detailed performance metrics

## Getting Started

# Local Deployment on single GPU
1. Create a conda environment and install dependencies:
```bash
conda create -n rag python=3.10 -y
conda activate rag
pip install -r task-2/requirements.txt
```

2. Download required models (if running on a GPU machine with limited internet access):
```bash
bashhuggingface-cli download intfloat/multilingual-e5-large-instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```

3. Run the RAG service:
```bash
cd task-2
python main.py
```

3. For load testing and performance evaluation:
```bash
cd task-2
bash benchmarks/load_test.sh
```

# Kubernetes-Based Deployment for autoscaling with Multiple GPUs
For containerized deployment with Kubernetes:
```bash
# Build container images
docker build -t rag-service:latest -f Dockerfile .
docker build -t rag-autoscaler:latest -f Dockerfile.autoscaler .

# Deploy to Kubernetes
kubectl apply -f deployment/redis.yaml
kubectl apply -f deployment/rag-service.yaml
kubectl apply -f deployment/rag-service-service.yaml
kubectl apply -f deployment/autoscaler.yaml
```
*Note:* autoscaler.py should be configured for preferences and also *automatic downscaling implementated* before using this deployment.

## Performance Evaluation

Our system has been extensively benchmarked under various load conditions to determine optimal batch sizes and scaling properties. Our testing has found the system to work optimally with a MAX_BATCH_SIZE=32 and a MAX_WAIT_TIME=1 for Nvidia Tesla L4 GPUs, however optimal settings may very on other GPUs/ CPUs. Important performance metrics include throughput, average latency, and P99 latency under different RPS (Requests Per Second) loads under different request patterns (random, poisson, uniform). 
