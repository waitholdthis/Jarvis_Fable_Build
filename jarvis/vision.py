"""Screen vision: capture the screen and describe it with a multimodal model.

Capture uses OS-native commands (no dependencies): screencapture on macOS,
gnome-screenshot/spectacle/grim/scrot on Linux, PowerShell on Windows. The
image is sent to the user's own local model as an OpenAI-style multimodal
message — Ollama and LM Studio both accept base64 data URLs when the loaded
model has vision (qwen2.5vl, llama3.2-vision, gemma3, llava, ...). If the
model can't see, the tool reports that plainly instead of pretending.
"""

from __future__ import annotations

import base64
import platform
import shutil
import subprocess
from pathlib import Path

DESCRIBE_PROMPT = (
    "This is a screenshot of the user's screen. Describe what is visible: "
    "the active application, any readable text or headings, and anything "
    "notable. Be concise and factual; do not guess at what you cannot see."
)


def screenshot_command(target: Path, system: str | None = None) -> list[str] | None:
    """Pick the native full-screen capture command for this OS, or None."""
    system = system or platform.system()
    path = str(target)
    if system == "Darwin":
        return ["screencapture", "-x", path]
    if system == "Linux":
        if shutil.which("gnome-screenshot"):
            return ["gnome-screenshot", "-f", path]
        if shutil.which("spectacle"):
            return ["spectacle", "-b", "-n", "-o", path]
        if shutil.which("grim"):  # wlroots Wayland
            return ["grim", path]
        if shutil.which("scrot"):  # X11
            return ["scrot", "-o", path]
        return None
    if system == "Windows":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
            "$b = [System.Windows.Forms.SystemInformation]::VirtualScreen;"
            "$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height;"
            "$g = [System.Drawing.Graphics]::FromImage($bmp);"
            "$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size);"
            f"$bmp.Save({path!r}, [System.Drawing.Imaging.ImageFormat]::Png)"
        )
        return ["powershell", "-NoProfile", "-Command", script]
    return None


def capture_screen(target: Path) -> bool:
    command = screenshot_command(target)
    if command is None:
        return False
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=20)
    except Exception:
        return False
    return target.is_file() and target.stat().st_size > 0


def image_message(prompt: str, image_path: Path) -> dict:
    """Build an OpenAI-style multimodal user message with an inline image."""
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"},
            },
        ],
    }


def describe_image(llm, image_path: Path, prompt: str = DESCRIBE_PROMPT) -> str:
    try:
        return llm.chat([image_message(prompt, image_path)], temperature=0.2)
    except Exception as exc:
        return (
            f"ERROR: the model could not process the image ({type(exc).__name__}). "
            f"Your current model '{getattr(llm, 'model', '?')}' may not support "
            "vision — try a multimodal one (e.g. `ollama pull qwen2.5vl:7b`)."
        )


def register_vision_tool(registry, llm) -> None:
    """Add the see_screen tool. Registered lazily because it needs the LLM."""
    from .tools import Tool

    config = registry.config

    def see_screen(question: str = "") -> str:
        shots = config.workspace / ".screenshots"
        shots.mkdir(parents=True, exist_ok=True)
        target = shots / "latest.png"
        if not capture_screen(target):
            return (
                "ERROR: no screenshot tool available on this system "
                "(need screencapture / gnome-screenshot / spectacle / grim / "
                "scrot / PowerShell)."
            )
        prompt = DESCRIBE_PROMPT
        if question:
            prompt += f"\nThe user specifically wants to know: {question}"
        return describe_image(llm, target, prompt)

    registry.register(
        Tool(
            name="see_screen",
            description=(
                "Take a screenshot of the user's screen and describe it "
                "(requires a vision-capable model)."
            ),
            params={"question": "optional: what to look for"},
            func=see_screen,
        )
    )
