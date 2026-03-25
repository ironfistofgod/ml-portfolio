FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY ds_config.json .

CMD ["accelerate", "launch", "--multi_gpu", "--num_processes=2", "train.py"]
