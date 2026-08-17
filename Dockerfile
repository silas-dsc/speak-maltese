# Speak Maltese — Hugging Face Space (Docker SDK), or any container host.
#
# Two things make the image behave on a free tier:
#
#   * The model is baked in at build time. Downloading 1.2GB of weights on first
#     request would put the whole wait on whoever opens the app after a restart,
#     and free Spaces restart often. Baking it costs image size and buys a boot
#     that only has to read from local disk.
#   * The dialogue audio is pre-rendered into the image too, so a scripted turn
#     never waits on text-to-speech and the app works with no outbound network.

FROM python:3.12-slim

# libgomp is required by onnxruntime/torch; ffmpeg decodes the browser's webm.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as uid 1000 and only that user's home is writable.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU-only torch. The default wheel pulls the entire CUDA stack — several GB of
# it — which no free tier has a GPU for and some have no room for.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

USER user
COPY --chown=user . .

# Fetch the Maltese recogniser at build time so the first request does not.
ARG SM_W2V_MODEL=carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h
ENV SM_W2V_MODEL=${SM_W2V_MODEL}
RUN python -c "\
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; \
import os; m = os.environ['SM_W2V_MODEL']; \
Wav2Vec2Processor.from_pretrained(m); Wav2Vec2ForCTC.from_pretrained(m); \
print('cached', m)"

# Render every line the app can speak. Needs network at build time (edge-tts);
# if it is unavailable the app still works, it just synthesises on demand.
RUN python scripts/prebuild_audio.py --what drills || \
    echo "audio prebuild skipped — will synthesise on demand"

# Named for the variable the app actually reads. It used to say SM_STT_CHAIN, which
# nothing looks at, so the chain stayed on `auto` — and `auto` ends in faster-whisper,
# whose weights are not baked into this image. The startup warm then downloaded a
# second recogniser at runtime, on a free tier, to back up one that was already
# loaded and 80x faster.
ENV SM_STT_PROVIDER=wav2vec2 \
    SM_PORT=7860
EXPOSE 7860

# One worker on purpose: the model is loaded per process and a free tier does not
# have the memory for two copies of it.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
