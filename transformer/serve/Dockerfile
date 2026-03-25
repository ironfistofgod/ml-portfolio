FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.py .
COPY serve.py .

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
