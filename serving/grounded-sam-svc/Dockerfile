# Base provides torch + CUDA. Build on RXL. GroundingDINO and segment-anything
# are installed here; their weights are mounted at runtime.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Phase 2 adds: pip install GroundingDINO + segment-anything and their deps.

COPY app ./app

ENV WORKPIECE_PROMPT="workpiece. metal part."
ENV SEG_DEVICE=cuda:0
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
