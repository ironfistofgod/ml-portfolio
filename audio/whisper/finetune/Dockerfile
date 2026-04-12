FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

ARG CACHEBUST=1
COPY finetune/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY finetune/train.py .

CMD ["python", "/app/train.py"]
