# TODO: base image, install F5-TTS from source, copy app.py + vocab.txt, expose 8000
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/SWivid/F5-TTS.git /opt/f5-tts && \
    pip install --no-cache-dir -e /opt/f5-tts && \
    pip uninstall flash-attn -y || true

ARG CACHEBUST=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vocab.txt .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]