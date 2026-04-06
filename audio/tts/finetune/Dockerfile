FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/SWivid/F5-TTS.git /opt/f5-tts && \
    pip install --no-cache-dir -e /opt/f5-tts

ARG CACHEBUST=1
COPY finetune/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_data.py ./prepare_data.py
COPY finetune/train.py .

CMD ["bash", "-c", "python /app/prepare_data.py && accelerate launch --mixed_precision bf16 /app/train.py"]