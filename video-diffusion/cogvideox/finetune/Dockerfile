FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall flash-attn -y || true

COPY train.py prepare_data.py ./

CMD ["python", "train.py"]
