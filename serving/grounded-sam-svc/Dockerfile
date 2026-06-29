# Base provides torch + CUDA + torchvision. Build on RXL (root-disk permitting).
# transformers (GroundingDINO) + segment-anything are installed from requirements.
# SAM weights are mounted at runtime; the GroundingDINO detector is downloaded by
# HF transformers to HF_HOME (mount a cache volume), via the HF mirror.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV WORKPIECE_PROMPT="workpiece. metal part."
ENV SEG_DEVICE=cuda:0
ENV GROUNDING_DINO_MODEL_ID="IDEA-Research/grounding-dino-tiny"
ENV HF_ENDPOINT="https://hf-mirror.com"
ENV HF_HOME="/ckpt/hf-cache"
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=40 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
