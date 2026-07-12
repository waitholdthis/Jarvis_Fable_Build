from pathlib import Path

from jarvis.config import Config
from jarvis.embeddings import HashEmbedder
from jarvis.memory import Memory
from jarvis.tools import ToolRegistry
from jarvis.vision import describe_image, image_message, register_vision_tool, screenshot_command

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepixels"


def test_screenshot_command_per_platform(monkeypatch, tmp_path):
    target = tmp_path / "s.png"
    assert screenshot_command(target, system="Darwin")[0] == "screencapture"
    assert screenshot_command(target, system="Windows")[0] == "powershell"

    import jarvis.vision as vision_mod
    monkeypatch.setattr(
        vision_mod.shutil, "which",
        lambda name: "/usr/bin/" + name if name == "grim" else None,
    )
    assert screenshot_command(target, system="Linux")[0] == "grim"
    monkeypatch.setattr(vision_mod.shutil, "which", lambda name: None)
    assert screenshot_command(target, system="Linux") is None


def test_image_message_builds_data_url(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(PNG_BYTES)
    message = image_message("what is this?", img)
    assert message["role"] == "user"
    text_part, image_part = message["content"]
    assert text_part == {"type": "text", "text": "what is this?"}
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert len(url) > 30


class VisionLLM:
    model = "fake-vl"

    def chat(self, messages, temperature=0.7):
        content = messages[0]["content"]
        assert any(part.get("type") == "image_url" for part in content)
        return "A code editor with a terminal at the bottom."


class BlindLLM:
    model = "text-only"

    def chat(self, messages, temperature=0.7):
        raise RuntimeError("400: model does not support images")


def test_describe_image_success_and_graceful_failure(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(PNG_BYTES)
    assert "code editor" in describe_image(VisionLLM(), img)
    failure = describe_image(BlindLLM(), img)
    assert failure.startswith("ERROR")
    assert "vision" in failure


def test_see_screen_tool_uses_capture_and_llm(tmp_path, monkeypatch):
    config = Config()
    config.home = tmp_path / "home"
    config.workspace = config.home / "workspace"
    config.ensure_dirs()
    memory = Memory(config.db_path, embedder=HashEmbedder())
    registry = ToolRegistry(config, memory)
    register_vision_tool(registry, VisionLLM())

    import jarvis.vision as vision_mod

    def fake_capture(target: Path) -> bool:
        target.write_bytes(PNG_BYTES)
        return True

    monkeypatch.setattr(vision_mod, "capture_screen", fake_capture)
    out = registry.run("see_screen", {"question": "which app is open?"})
    assert "code editor" in out

    monkeypatch.setattr(vision_mod, "capture_screen", lambda target: False)
    assert registry.run("see_screen", {}).startswith("ERROR: no screenshot tool")
