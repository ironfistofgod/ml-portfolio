FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_data.py .
COPY train.py .
COPY fsdp_config.yaml .

CMD ["accelerate", "launch", "--config_file", "fsdp_config.yaml", "train.py"]
