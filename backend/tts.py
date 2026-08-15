"""Maltese text-to-speech with a fallback chain.

Maltese TTS is genuinely scarce, so the ordering here matters:

* **Azure** — the only major cloud with real `mt-MT` neural voices
  (`mt-MT-GraceNeural`, `mt-MT-JosephNeural`). Needs a key. Best quality, SSML rate
  control, and a licence that covers this use.
* **edge** — Microsoft Edge's Read-Aloud endpoint exposes those *same two voices*
  with no key at all, which is what makes this app usable on a fresh clone. It is an
  unofficial endpoint (`edge-tts`), so it is not for anything commercial and it can
  break without warning — hence Azure first when a key exists.
* **ElevenLabs** — v3 covers Maltese; useful if you already pay for it.
* **mms** — Meta's `facebook/mms-tts-mlt` VITS model, fully offline. Robotic but
  phonetically sound. Opt-in because it drags in torch. Set `SM_TTS_PROVIDER=mms`.

Deliberately *not* used: browser `speechSynthesis` (no OS ships a Maltese voice, and
an Italian voice reading Maltese teaches wrong pronunciation) and gTTS (Google
Translate has no Maltese audio — it errors with "Unsupported language 'mt'").
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging

import httpx

from .config import CFG, AUDIO_CACHE

log = logging.getLogger("speak-maltese.tts")

_mms = None  # lazily loaded (model, tokenizer)


class TTSError(RuntimeError):
    pass


def _cache_key(text: str, voice: str, rate: float, provider: str) -> str:
    return hashlib.sha256(f"{provider}|{voice}|{rate}|{text}".encode()).hexdigest()[:32]


async def synthesize(text: str, voice: str | None = None,
                     rate: float = 1.0) -> tuple[bytes, str]:
    """Return (audio_bytes, mime). Tries each configured provider in order."""
    text = (text or "").strip()
    if not text:
        raise TTSError("empty text")

    errors: list[str] = []
    for provider in CFG.tts_chain():
        v = _resolve_voice(provider, voice)
        ext = "wav" if provider == "mms" else "mp3"
        cached = AUDIO_CACHE / f"{_cache_key(text, v, rate, provider)}.{ext}"
        mime = "audio/wav" if provider == "mms" else "audio/mpeg"
        if cached.exists() and cached.stat().st_size > 0:
            return cached.read_bytes(), mime
        try:
            audio = await _dispatch(provider, text, v, rate)
            if audio:
                cached.write_bytes(audio)
                return audio, mime
        except Exception as exc:  # noqa: BLE001 — one provider failing is not fatal
            log.warning("TTS provider %s failed: %s", provider, exc)
            errors.append(f"{provider}: {exc}")
    raise TTSError("; ".join(errors) or "no TTS provider available")


def _resolve_voice(provider: str, requested: str | None) -> str:
    """Voice names are shared between azure and edge; fall back per provider."""
    if provider in ("azure", "edge"):
        if requested and requested.startswith("mt-MT-"):
            return requested
        return CFG.azure_voice
    if provider == "elevenlabs":
        return requested if (requested and not requested.startswith("mt-")) else CFG.elevenlabs_voice
    return "mlt"


async def _dispatch(provider: str, text: str, voice: str, rate: float) -> bytes:
    if provider == "azure":
        return await _azure(text, voice, rate)
    if provider == "edge":
        return await _edge(text, voice, rate)
    if provider == "elevenlabs":
        return await _elevenlabs(text, voice)
    if provider == "mms":
        return await asyncio.to_thread(_mms_tts, text, rate)
    raise TTSError(f"unknown TTS provider {provider!r}")


async def _azure(text: str, voice: str, rate: float) -> bytes:
    if not CFG.azure_speech_key:
        raise TTSError("AZURE_SPEECH_KEY not set")
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="mt-MT">'
        f'<voice name="{html.escape(voice)}">'
        f'<prosody rate="{_rate_pct(rate)}">{html.escape(text)}</prosody>'
        "</voice></speak>"
    )
    url = f"https://{CFG.azure_speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": CFG.azure_speech_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
                "User-Agent": "speak-maltese",
            },
            content=ssml.encode("utf-8"),
        )
        r.raise_for_status()
        return r.content


async def _edge(text: str, voice: str, rate: float) -> bytes:
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=_rate_pct(rate))
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    if not buf:
        raise TTSError("edge returned no audio")
    return bytes(buf)


async def _elevenlabs(text: str, voice: str) -> bytes:
    if not (CFG.elevenlabs_key and voice):
        raise TTSError("ELEVENLABS_KEY / SM_ELEVENLABS_VOICE not set")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
            headers={"xi-api-key": CFG.elevenlabs_key, "content-type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_v3",
                "language_code": "mt",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        r.raise_for_status()
        return r.content


def _mms_tts(text: str, rate: float) -> bytes:
    """Meta MMS (VITS) — offline Maltese. First call downloads ~145 MB."""
    global _mms
    import io
    import wave

    import numpy as np

    if _mms is None:
        from transformers import VitsModel, AutoTokenizer

        log.info("loading facebook/mms-tts-mlt (first run downloads the model)")
        model = VitsModel.from_pretrained("facebook/mms-tts-mlt")
        tok = AutoTokenizer.from_pretrained("facebook/mms-tts-mlt")
        _mms = (model, tok)

    import torch

    model, tok = _mms
    model.speaking_rate = max(0.5, min(1.6, rate))
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        wave_f = model(**inputs).waveform[0].cpu().numpy()

    pcm = (np.clip(wave_f, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(model.config.sampling_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _rate_pct(rate: float) -> str:
    return f"{int(round((rate - 1.0) * 100)):+d}%"


def available() -> list[str]:
    out = []
    for p in CFG.tts_chain():
        if p == "azure" and CFG.azure_speech_key:
            out.append(p)
        elif p == "edge":
            try:
                import edge_tts  # noqa: F401
                out.append(p)
            except ImportError:
                pass
        elif p == "elevenlabs" and CFG.elevenlabs_key and CFG.elevenlabs_voice:
            out.append(p)
        elif p == "mms":
            try:
                import transformers  # noqa: F401
                out.append(p)
            except ImportError:
                pass
    return out
