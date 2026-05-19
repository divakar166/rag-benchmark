import modal
import os

# Config
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_DIR  = "/model-cache/models"
MODEL_PATH = f"{MODEL_DIR}/Qwen2.5-Coder-7B-Instruct"

app = modal.App("vllm-qwen")

model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
vllm_secret = modal.Secret.from_name("vllm-secrets")

# Image: lightweight downloader
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.24.6")  # hf_transfer = fast parallel download
    .env({
        "HF_HOME": "/model-cache/huggingface",
        "HF_HUB_CACHE": "/model-cache/huggingface/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",               # enables fast download
    })
)

# Image: CUDA devel for vLLM serving
serve_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
        force_build=True,
    )
    .apt_install("git", "build-essential", "curl")
    .pip_install(
        "vllm==0.6.3.post1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface_hub==0.24.6",
        "fastapi",
        "uvicorn",
        "outlines[all]",
        "pyairports",
    )
    .env({
        "HF_HOME": "/model-cache/huggingface",
        "HF_HUB_CACHE": "/model-cache/huggingface/hub",

        # Disable FlashInfer JIT to prevent compile failures on cold start
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "FLASHINFER_DISABLE_JIT": "1",

        # Disable torch compile to reduce cold start time
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    })
)

# Step 1: Download model to Volume
# Run once manually: modal run modal_vllm_qwen.py::download_model
@app.function(
    image=download_image,
    secrets=[vllm_secret],
    volumes={"/model-cache": model_cache},
    timeout=60 * 45,
)
def download_model():
    from huggingface_hub import snapshot_download
    import os

    hf_token = os.environ.get("HF_TOKEN")

    print(f"Downloading {MODEL_NAME} → {MODEL_PATH}")

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_PATH,
        ignore_patterns=["*.pt", "*.bin"],
        token=hf_token,
    )

    model_cache.commit()
    print(f"Model committed to Modal Volume at {MODEL_PATH}")


# Step 2: Serve via vLLM OpenAI-compatible API
# Deploy: modal deploy modal_vllm_qwen.py
@app.function(
    image=serve_image,
    gpu="A10G",
    secrets=[vllm_secret],
    volumes={"/model-cache": model_cache},
    timeout=60 * 60,          # 1 hour max request lifetime
    scaledown_window=300,     # keep warm for 5 min after last request
    max_containers=1,
)
@modal.web_server(port=8000, startup_timeout=60 * 10)  # 10 min is enough for 7B
def serve():
    import subprocess
    import time

    model_cache.reload()

    import os
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run `modal run modal_vllm_qwen.py::download_model` first."
        )

    print(f"Starting vLLM server for {MODEL_NAME}...")

    api_key = os.environ["VLLM_API_KEY"]

    process = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",

            "--host", "0.0.0.0",
            "--port", "8000",

            "--model", MODEL_PATH,
            "--served-model-name", MODEL_NAME,

            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",

            # enforce-eager disables CUDA graph capture → faster cold start,
            # slightly lower steady-state throughput. Remove for production.
            "--enforce-eager",

            # Allows multiple concurrent requests to be batched
            "--max-num-seqs", "16",

            "--api-key", api_key,
        ],
        # Pipe output so Modal captures vLLM logs
        stdout=None,  # inherit → shows in modal logs
        stderr=None,
    )

    # Give vLLM a moment to bind the port before Modal's health probe hits it
    time.sleep(5)

    # If the process died immediately, surface the error
    ret = process.poll()
    if ret is not None:
        raise RuntimeError(f"vLLM process exited immediately with code {ret}")

    print("vLLM server started. Waiting for requests...")