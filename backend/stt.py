"""Maltese speech-to-text with a fallback chain.

Browser `SpeechRecognition` cannot be used — no browser ships a Maltese acoustic
model — so audio is recorded in the page and posted here.

Provider notes:
* **OpenAI Whisper** (`whisper-1`) — Maltese is in its language set; accepts webm/opus
  straight from MediaRecorder. Best accuracy-per-effort with a key.
* **ElevenLabs Scribe** — strong on Maltese, also accepts webm.
* **faster-whisper** — runs locally with no key, decodes webm via PyAV. Slower on
  first use (model download) but keeps the app fully offline-capable. `SM_WHISPER_MODEL`
  takes a plain size *or* any CTranslate2 repo on the Hub, so it can load a
  Maltese-fine-tuned Whisper instead of the generic multilingual one — worth doing,
  since generic Whisper is weak on a language with this little training data.
* **Azure** — good, but its short-audio REST endpoint wants WAV/OGG-Opus, so it is
  tried last and only when ffmpeg is available for transcoding.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from .config import CFG

log = logging.getLogger("speak-maltese.stt")

_whisper_model = None


class STTError(RuntimeError):
    pass


async def transcribe(audio: bytes, mime: str = "audio/webm") -> dict:
    """Return {"text": str, "provider": str}."""
    if not audio:
        raise STTError("empty audio")
    errors: list[str] = []
    for provider in CFG.stt_chain():
        try:
            text = await _dispatch(provider, audio, mime)
            if text is not None:
                return {"text": text.strip(), "provider": provider}
        except Exception as exc:  # noqa: BLE001
            log.warning("STT provider %s failed: %s", provider, exc)
            errors.append(f"{provider}: {exc}")
    raise STTError("; ".join(errors) or "no STT provider available")


async def _dispatch(provider: str, audio: bytes, mime: str) -> str | None:
    if provider == "openai_whisper":
        return await _openai(audio, mime)
    if provider == "elevenlabs":
        return await _elevenlabs(audio, mime)
    if provider == "wav2vec2":
        return await asyncio.to_thread(_wav2vec2, audio, mime)
    if provider == "faster_whisper":
        return await asyncio.to_thread(_faster_whisper, audio, mime)
    if provider == "azure":
        return await _azure(audio, mime)
    raise STTError(f"unknown STT provider {provider!r}")


def _ext(mime: str) -> str:
    return {
        "audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4",
        "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
    }.get(mime.split(";")[0].strip(), "webm")


async def _openai(audio: bytes, mime: str) -> str:
    if not CFG.openai_key:
        raise STTError("OPENAI_API_KEY not set")
    files = {"file": (f"speech.{_ext(mime)}", audio, mime)}
    data = {"model": "whisper-1", "language": "mt", "response_format": "json"}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"authorization": f"Bearer {CFG.openai_key}"},
            files=files, data=data,
        )
        r.raise_for_status()
        return r.json().get("text", "")


async def _elevenlabs(audio: bytes, mime: str) -> str:
    if not CFG.elevenlabs_key:
        raise STTError("ELEVENLABS_KEY not set")
    files = {"file": (f"speech.{_ext(mime)}", audio, mime)}
    data = {"model_id": "scribe_v1", "language_code": "mlt"}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": CFG.elevenlabs_key},
            files=files, data=data,
        )
        r.raise_for_status()
        return r.json().get("text", "")


_w2v = None


def _load_wav2vec2():
    """Maltese wav2vec2 CTC, on Apple Silicon's GPU where available.

    This is the fast path, and the reason is structural rather than a matter of
    model size. Whisper is autoregressive and pads every input to a fixed 30-second
    window, so a three-word answer costs the same as a monologue. A CTC model is a
    single forward pass over the audio you actually recorded — no decoder loop, no
    padding — which on a 2-second utterance is the difference between ~8s and ~0.1s
    at the same accuracy.

    Output is lowercase and unpunctuated. That is fine here: the drill matches
    phonetically, and the tutor reads it as input rather than showing it.
    """
    global _w2v
    if _w2v is not None:
        return _w2v
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = CFG.w2v_device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("loading wav2vec2 %s on %s", CFG.w2v_model, device)
    t0 = time.time()
    processor = Wav2Vec2Processor.from_pretrained(CFG.w2v_model)
    model = Wav2Vec2ForCTC.from_pretrained(CFG.w2v_model).to(device).eval()
    log.info("wav2vec2 ready in %.1fs", time.time() - t0)
    _w2v = (processor, model, device)
    return _w2v


def _wav2vec2(audio: bytes, mime: str) -> str:
    import torch
    from faster_whisper.audio import decode_audio  # PyAV: handles webm/opus

    processor, model, device = _load_wav2vec2()

    with tempfile.NamedTemporaryFile(suffix=f".{_ext(mime)}", delete=False) as fh:
        fh.write(audio)
        path = fh.name
    try:
        wave = decode_audio(path, sampling_rate=16000)
        inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(ids)[0]
    finally:
        Path(path).unlink(missing_ok=True)


def needs_warmup() -> bool:
    """Is there a local model in the chain that has to be loaded before use?

    A hosted-API chain (ElevenLabs, Azure) is ready the moment the process is up;
    a local model is not, and on a cold free-tier container the load is tens of
    seconds. The client uses this to decide whether to show a startup screen at all
    rather than flashing one for a deployment that never needed it."""
    return bool({"wav2vec2", "faster_whisper"} & set(available()))


def is_warm() -> bool:
    """Has a local model actually finished loading? Reports the *loaded* object,
    not the fact that a preload thread was started — the client waits on this, so
    an optimistic answer would put the wait back on the first utterance."""
    return _w2v is not None or _whisper_model is not None


def preload() -> None:
    """Load the local model now rather than on the learner's first sentence.

    The Maltese fine-tune is a whisper-large, so a cold load costs several seconds.
    Paying that at boot means the first thing you say is answered as fast as the
    tenth. Safe to call when faster-whisper is not the active backend — it simply
    warms a model that may go unused.
    """
    chain = CFG.stt_chain()
    for loader, name in ((_load_wav2vec2, "wav2vec2"), (_load_whisper, "faster_whisper")):
        if name not in chain:
            continue
        try:
            loader()
        except Exception as exc:  # noqa: BLE001 — warming is best-effort
            log.warning("could not preload %s: %s", name, exc)


def _load_whisper():
    global _whisper_model
    from faster_whisper import WhisperModel

    if _whisper_model is None:
        device = CFG.whisper_device
        if device == "auto":
            device = "cpu"
        compute = "int8" if device == "cpu" else "float16"
        log.info("loading faster-whisper %s on %s", CFG.whisper_model, device)
        t0 = time.time()
        _whisper_model = WhisperModel(CFG.whisper_model, device=device, compute_type=compute)
        log.info("local STT ready in %.1fs", time.time() - t0)
    return _whisper_model


def _faster_whisper(audio: bytes, mime: str) -> str:
    model = _load_whisper()

    with tempfile.NamedTemporaryFile(suffix=f".{_ext(mime)}", delete=False) as fh:
        fh.write(audio)
        path = fh.name
    try:
        segments, _info = model.transcribe(
            path, language="mt", beam_size=5,
            vad_filter=True,
            # Pad the detected speech region so VAD cannot shave the onset or tail
            # off a short utterance — a lost initial consonant turns "Bonġu" into
            # "onġu". Measured no accuracy cost on clean audio.
            vad_parameters={"speech_pad_ms": 400},
        )
        return " ".join(s.text for s in segments)
    finally:
        Path(path).unlink(missing_ok=True)


async def _azure(audio: bytes, mime: str) -> str:
    if not CFG.azure_speech_key:
        raise STTError("AZURE_SPEECH_KEY not set")
    wav = _to_wav(audio, mime)
    url = (
        f"https://{CFG.azure_speech_region}.stt.speech.microsoft.com"
        "/speech/recognition/conversation/cognitiveservices/v1?language=mt-MT"
    )
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": CFG.azure_speech_key,
                "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                "Accept": "application/json",
            },
            content=wav,
        )
        r.raise_for_status()
        body = r.json()
    if body.get("RecognitionStatus") != "Success":
        raise STTError(f"azure status {body.get('RecognitionStatus')}")
    return body.get("DisplayText", "")


def _to_wav(audio: bytes, mime: str) -> bytes:
    if _ext(mime) == "wav":
        return audio
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise STTError("ffmpeg not found; cannot transcode for Azure")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        input=audio, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise STTError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


def available() -> list[str]:
    out = []
    for p in CFG.stt_chain():
        if p == "openai_whisper" and CFG.openai_key:
            out.append(p)
        elif p == "elevenlabs" and CFG.elevenlabs_key:
            out.append(p)
        elif p == "azure" and CFG.azure_speech_key and shutil.which("ffmpeg"):
            out.append(p)
        elif p == "wav2vec2":
            try:
                import transformers, torch  # noqa: F401
                out.append(p)
            except ImportError:
                pass
        elif p == "faster_whisper":
            try:
                import faster_whisper  # noqa: F401
                out.append(p)
            except ImportError:
                pass
    return out
