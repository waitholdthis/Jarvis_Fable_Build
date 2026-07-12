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


def listen_until_silence(
    max_seconds: float = 30.0,
    silence_after: float = 1.2,
    energy_threshold: float = 0.012,
    model_size: str = "base",
) -> str:
    """Hands-free capture: wait for speech, record until the user goes quiet.

    A simple RMS-energy VAD — no extra dependencies beyond the [voice] extra.
    Returns the transcript, or "" if nothing was heard within max_seconds.
    """
    import numpy as np
    import sounddevice as sd

    sample_rate = 16_000
    block = int(sample_rate * 0.05)  # 50 ms frames
    silence_blocks_needed = int(silence_after / 0.05)
    max_blocks = int(max_seconds / 0.05)

    frames: list = []
    speech_started = False
    silent_blocks = 0

    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32", blocksize=block
    ) as stream:
        for _ in range(max_blocks):
            data, _overflow = stream.read(block)
            rms = float(np.sqrt(np.mean(np.square(data))))
            if not speech_started:
                if rms >= energy_threshold:
                    speech_started = True
                    frames.append(data.copy())
                continue
            frames.append(data.copy())
            silent_blocks = silent_blocks + 1 if rms < energy_threshold else 0
            if silent_blocks >= silence_blocks_needed:
                break

    if not speech_started:
        return ""
    audio = np.squeeze(np.concatenate(frames))
    model = _get_whisper(model_size)
    segments, _info = model.transcribe(audio, beam_size=1, vad_filter=True)
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
