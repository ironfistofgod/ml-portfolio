FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall diffusers==0.31.0 && \
    pip uninstall flash-attn -y || true

COPY train.py prepare_data.py ./

CMD ["bash", "-c", "python /app/prepare_data.py && python /app/train.py"]
