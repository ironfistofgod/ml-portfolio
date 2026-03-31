FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Clone finetrainers and install its dependencies
RUN git clone https://github.com/huggingface/finetrainers /workspace/finetrainers
RUN pip install --no-cache-dir -r /workspace/finetrainers/requirements.txt

# Diffusers from source — required for latest Wan2.1 support
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers

# Extra tools
RUN pip install --no-cache-dir wandb huggingface_hub

COPY train.py training.json validation.json ./

CMD ["python", "train.py"]
