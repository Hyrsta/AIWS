FROM cadrille:latest

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# run_cadrille.py is the thin adapter CLI that wraps the existing cadrille
# wrappers and emits the canonical bundle layout. It is provided by the AIWS
# integration repo and copied to CADRILLE_ENTRY at build time on RXL.
COPY run_cadrille.py /opt/aiws/run_cadrille.py

ENV CADRILLE_ENTRY=/opt/aiws/run_cadrille.py
ENV CADRILLE_DEVICE=cuda:0
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
