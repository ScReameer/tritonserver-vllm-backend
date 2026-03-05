FROM nvcr.io/nvidia/tritonserver:26.02-vllm-python-py3

ARG BASE_PATH="/runpod-volume"

ENV BASE_PATH="${BASE_PATH}" \
    HF_HOME="${BASE_PATH}/huggingface-cache" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    VLLM_CACHE_ROOT="${BASE_PATH}/vllm-cache"

COPY model_repository /models

COPY openai_frontend /opt/tritonserver/python/openai/openai_frontend
COPY vllm_backend /opt/tritonserver/backends/vllm

# For Runpod
EXPOSE 80

CMD ["bash", "-lc", "python3 /opt/tritonserver/python/openai/openai_frontend/main.py --model-repository /models --openai-port ${PORT}"]
