FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/hf_cache \
    MOCK_MODEL=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# bake the three backbones into the image so the first request doesn't download ~2 GB
RUN python -c "from transformers import AutoModel, CLIPModel; AutoModel.from_pretrained('facebook/dinov2-base'); AutoModel.from_pretrained('facebook/dinov2-large'); CLIPModel.from_pretrained('openai/clip-vit-base-patch16')"

COPY . /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]