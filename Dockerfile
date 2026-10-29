# syntax=docker/dockerfile:1
# recsterio backend image — used by both the API service and the worker service.
# One image, two commands: `uvicorn app.main:app` (API) and `python -m app.worker`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg is a system binary the ASR path depends on (16 kHz mono WAV extraction);
# it is NOT a pip package.
# nodejs runs yt-dlp's EJS challenge-solver for YouTube — without it YouTube
# serves no media formats.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Heavy ML deps (faster-whisper, google-genai) plus the core requirements.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY app ./app

# Writable state, all on named volumes so it survives redeploys:
#   /data/media  - downloaded media + extracted audio
#   /data/state  - cross-process worker heartbeat
#   /data/models - Whisper model cache (re-downloaded otherwise, ~460 MB for `small`)
RUN mkdir -p /data/media /data/state /data/models /data/cookies && chown -R app:app /data
USER app

ENV MEDIA_ROOT=/data/media \
    WORKER_HEARTBEAT_FILE=/data/state/worker_heartbeat.json \
    HF_HOME=/data/models \
    WORKER_AUTOSTART=false \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
