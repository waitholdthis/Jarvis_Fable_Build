"""Optional voice I/O. Everything degrades gracefully.

ASR: faster-whisper + sounddevice (installed via the [voice] extra).
TTS: pyttsx3 if installed, else the platform's native speech command
     (`say` on macOS, `espeak`/`spd-say` on Linux, PowerShell on Windows),
     else silence. No import here is mandatory: on a machine with no audio
     stack Jarvis simply stays a text assistant.
"""

from __future__ import annotations

import platform
import shutil
import subprocess


def asr_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        import sounddevice  # noqa: F401
        return True
    except Exception:
        return False


def record_and_transcribe(seconds: float = 6.0, model_size: str = "base") -> str:
    """Record from the default microphone and return the transcript."""
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel

    sample_rate = 16_000
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    model = _get_whisper(model_size)
    segments, _info = model.transcribe(np.squeeze(audio), beam_size=1, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


_whisper_cache: dict[str, object] = {}


def _get_whisper(model_size: str):
    if model_size not in _whisper_cache:
        from faster_whisper import WhisperModel

        _whisper_cache[model_size] = WhisperModel(
            model_size, device="auto", compute_type="int8"
        )
    return _whisper_cache[model_size]


def speak(text: str) -> bool:
    """Speak text with the best available backend. Returns True if spoken."""
    text = text.strip()
    if not text:
        return False

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        pass

    system = platform.system()
    commands: list[list[str]] = []
    if system == "Darwin":
        commands.append(["say", text])
    elif system == "Linux":
        if shutil.which("spd-say"):
            commands.append(["spd-say", "--wait", text])
        if shutil.which("espeak"):
            commands.append(["espeak", text])
    elif system == "Windows":
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
            f".Speak({text!r})"
        )
        commands.append(["powershell", "-NoProfile", "-Command", script])

    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            return True
        except Exception:
            continue
    return False
