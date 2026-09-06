# Deploy do app (oraculo + mercado + web) rodando na VPS/Coolify.
# CPU-only (sem GPU): usa torch CPU + yolo11n em baixa resolucao.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CPU (mais leve que a versao CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY . .

EXPOSE 8000
CMD ["python", "server.py", "--config", "config.docker.yaml", "--port", "8000"]
