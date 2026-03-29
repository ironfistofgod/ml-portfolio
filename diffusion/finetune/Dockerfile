FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_style.py .
COPY train_dreambooth.py .


ENV HF_HOME=/workspace/models
ENV WANDB_SILENT=true

ENTRYPOINT ["accelerate", "launch", "--num_processes=1", "train_style.py"]