from jarvis.config import Config
from jarvis.llm import NoRuntimeError, detect, pick_model


def test_pick_model_prefers_instruct_generalists():
    models = ["bge-m3", "qwen2.5:7b-instruct", "whisper"]
    assert pick_model(models) == "qwen2.5:7b-instruct"


def test_pick_model_honors_exact_and_prefix_request():
    models = ["llama3.2:3b", "qwen2.5:7b"]
    assert pick_model(models, "qwen2.5:7b") == "qwen2.5:7b"
    assert pick_model(models, "llama3.2") == "llama3.2:3b"


def test_pick_model_trusts_unlisted_request():
    # Ollama lazy-loads models not in /v1/models; trust the user's choice.
    assert pick_model(["other"], "mistral:7b") == "mistral:7b"


def test_pick_model_empty():
    assert pick_model([]) == ""


def test_detect_raises_helpful_error_when_nothing_running(monkeypatch):
    import jarvis.llm as llm_mod

    def refuse(base, timeout=0.6):
        raise ConnectionError("refused")

    monkeypatch.setattr(llm_mod, "_probe", refuse)
    config = Config()
    try:
        detect(config)
        raise AssertionError("expected NoRuntimeError")
    except NoRuntimeError as exc:
        assert "Ollama" in str(exc)
        assert "JARVIS_API_BASE" in str(exc)


def test_detect_uses_configured_endpoint_first(monkeypatch):
    import jarvis.llm as llm_mod

    def probe(base, timeout=0.6):
        if base == "http://myserver:9000/v1":
            return ["my-model"]
        raise ConnectionError("refused")

    monkeypatch.setattr(llm_mod, "_probe", probe)
    config = Config()
    config.api_base = "http://myserver:9000/v1"
    client = detect(config)
    assert client.model == "my-model"
    assert client.runtime_name == "configured endpoint"
