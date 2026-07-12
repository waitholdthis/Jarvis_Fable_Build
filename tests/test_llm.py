from jarvis.config import Config
from jarvis.llm import AnthropicClient, GeminiClient, NoRuntimeError, cloud_client, detect, pick_model


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


def test_cloud_provider_clients_use_environment_keys(monkeypatch):
    config = Config()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    assert isinstance(cloud_client(config, "claude"), AnthropicClient)
    assert isinstance(cloud_client(config, "gemini"), GeminiClient)
    assert cloud_client(config, "perplexity").api_base.endswith("/v1")
    assert cloud_client(config, "codex").runtime_name == "Codex"


def test_cloud_provider_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        cloud_client(Config(), "claude")
        raise AssertionError("expected missing-key error")
    except NoRuntimeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)


def test_detect_uses_selected_cloud_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    config = Config(provider="gemini", model="gemini-custom")
    client = detect(config)
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-custom"
