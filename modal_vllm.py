import modal

# Config
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "main"

VLLM_PORT = 8000
MINUTES = 60  # seconds

app = modal.App("vllm-qwen")

# Volumes — HF model cache + vLLM compile cache
# Both persist across cold starts
hf_cache_vol   = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
vllm_secret    = modal.Secret.from_name("vllm-secrets")

# Image
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.19.0",
        "transformers",
        "tokenizers",
        "huggingface_hub",
    )
    .env({
        # Model cache paths
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_CACHE": "/root/.cache/huggingface/hub",

        # Faster model transfers from HuggingFace
        "HF_XET_HIGH_PERFORMANCE": "1",

        # Disable FlashInfer JIT — prevents compile failures on cold start
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "FLASHINFER_DISABLE_JIT": "1",
    })
)

# Serve
@app.function(
    image=vllm_image,
    gpu="A10G",
    secrets=[vllm_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,   # model weights persist here
        "/root/.cache/vllm": vllm_cache_vol,        # compiled kernels persist here
    },
    timeout=5 * MINUTES,            # container startup timeout
    scaledown_window=1 * MINUTES,   # stay warm 5 min after last request
    max_containers=1,
)
@modal.concurrent(max_inputs=32)    # how many requests one replica handles concurrently
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import os
    import subprocess

    api_key = os.environ["VLLM_API_KEY"]

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,

        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),

        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",

        "--api-key", api_key,
    ]

    print("Starting vLLM:", " ".join(cmd))
    subprocess.Popen(cmd)