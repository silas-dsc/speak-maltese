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


def _faster_whisper(audio: bytes, mime: str) -> str:
    global _whisper_model
    from faster_whisper import WhisperModel

    if _whisper_model is None:
        device = CFG.whisper_device
        if device == "auto":
            device = "cpu"
        compute = "int8" if device == "cpu" else "float16"
        log.info("loading faster-whisper %s on %s", CFG.whisper_model, device)
        _whisper_model = WhisperModel(CFG.whisper_model, device=device, compute_type=compute)

    with tempfile.NamedTemporaryFile(suffix=f".{_ext(mime)}", delete=False) as fh:
        fh.write(audio)
        path = fh.name
    try:
        segments, _info = _whisper_model.transcribe(
            path, language="mt", vad_filter=True, beam_size=5,
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
        elif p == "faster_whisper":
            try:
                import faster_whisper  # noqa: F401
                out.append(p)
            except ImportError:
                pass
    return out
