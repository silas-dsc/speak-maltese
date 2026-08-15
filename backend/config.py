"""Central config. Everything is env-driven and every provider is optional —
the app degrades gracefully rather than refusing to start."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"
DB_PATH = Path(os.getenv("SM_DB_PATH", ROOT / "data" / "progress.db"))
AUDIO_CACHE = Path(os.getenv("SM_AUDIO_CACHE", ROOT / "data" / "audio_cache"))


def _b(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # ── LLM tutor ──────────────────────────────────────────────────────────
    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    tutor_model: str = field(default_factory=lambda: os.getenv("SM_TUTOR_MODEL", "claude-opus-5"))
    # Any OpenAI-compatible endpoint (Ollama, LM Studio, OpenRouter) as an alternative.
    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base: str = field(default_factory=lambda: os.getenv("SM_OPENAI_BASE_URL", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("SM_OPENAI_MODEL", "gpt-4o-mini"))

    # ── Text to speech ─────────────────────────────────────────────────────
    # Azure is the only major cloud with genuine mt-MT neural voices.
    azure_speech_key: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_KEY", ""))
    azure_speech_region: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_REGION", "westeurope"))
    azure_voice: str = field(default_factory=lambda: os.getenv("SM_AZURE_VOICE", "mt-MT-GraceNeural"))
    azure_voice_alt: str = field(default_factory=lambda: os.getenv("SM_AZURE_VOICE_ALT", "mt-MT-JosephNeural"))
    elevenlabs_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_KEY", ""))
    elevenlabs_voice: str = field(default_factory=lambda: os.getenv("SM_ELEVENLABS_VOICE", ""))
    tts_provider: str = field(default_factory=lambda: os.getenv("SM_TTS_PROVIDER", "auto"))

    # ── Speech to text ─────────────────────────────────────────────────────
    stt_provider: str = field(default_factory=lambda: os.getenv("SM_STT_PROVIDER", "auto"))
    whisper_model: str = field(default_factory=lambda: os.getenv("SM_WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: os.getenv("SM_WHISPER_DEVICE", "auto"))

    # ── Learning behaviour ─────────────────────────────────────────────────
    daily_new_limit: int = field(default_factory=lambda: int(os.getenv("SM_DAILY_NEW", "12")))
    daily_review_limit: int = field(default_factory=lambda: int(os.getenv("SM_DAILY_REVIEWS", "150")))
    target_retention: float = field(default_factory=lambda: float(os.getenv("SM_TARGET_RETENTION", "0.9")))
    host: str = field(default_factory=lambda: os.getenv("SM_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("SM_PORT", "8137")))
    debug: bool = field(default_factory=lambda: _b("SM_DEBUG"))

    # ── Capability report, surfaced in the UI ──────────────────────────────
    def capabilities(self) -> dict:
        return {
            "tutor": bool(self.anthropic_key) or bool(self.openai_key) or bool(self.openai_base),
            "tutor_provider": (
                "anthropic" if self.anthropic_key
                else "openai" if (self.openai_key or self.openai_base)
                else None
            ),
            "tts": self.tts_chain(),
            "stt": self.stt_chain(),
        }

    def tts_chain(self) -> list[str]:
        """Ordered list of TTS backends we can actually try."""
        if self.tts_provider != "auto":
            return [self.tts_provider]
        chain: list[str] = []
        if self.azure_speech_key:
            chain.append("azure")
        chain.append("edge")  # same mt-MT neural voices, no key required
        if self.elevenlabs_key and self.elevenlabs_voice:
            chain.append("elevenlabs")
        return chain

    def stt_chain(self) -> list[str]:
        if self.stt_provider != "auto":
            return [self.stt_provider]
        chain: list[str] = []
        if self.openai_key:
            chain.append("openai_whisper")
        if self.elevenlabs_key:
            chain.append("elevenlabs")
        chain.append("faster_whisper")  # local fallback, no key
        if self.azure_speech_key:
            # last: its short-audio endpoint needs a WAV transcode via ffmpeg
            chain.append("azure")
        return chain


CFG = Config()

AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
