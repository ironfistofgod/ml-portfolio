FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall diffusers==0.31.0 transformers==4.44.2 && \
    pip uninstall flash-attn -y || true

COPY app.py .

EXPOSE 8000

# ENTRYPOINT overrides base image — use ENTRYPOINT not CMD (see LESSONS_LEARNED)
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
