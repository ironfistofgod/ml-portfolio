FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN python3 -m pip install --no-cache-dir awscli

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    pip uninstall flash-attn -y || true

ENV HF_HOME=/workspace/models
ENV WANDB_SILENT=true

CMD ["sleep", "infinity"]