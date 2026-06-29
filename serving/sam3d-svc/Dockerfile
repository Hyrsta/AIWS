# Base image is the SAM3D conda environment. Build this base on RXL from the
# existing SAM3D environment, with the MKL fix baked in: the environment must
# NOT carry the base-conda sequential MKL via LD_PRELOAD. The verification step
# below fails the build if a leaked LD_PRELOAD is present.
FROM aiws-sam3d-base:latest

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY run_sam3d.py /opt/aiws/run_sam3d.py

# Verify no leaked MKL preload is baked into the image environment.
RUN test -z "${LD_PRELOAD}" || (echo "LD_PRELOAD must be empty in sam3d image" && exit 1)

ENV SAM3D_ENTRY=/opt/aiws/run_sam3d.py
ENV SAM3D_DEVICE=cuda:0
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=20 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
