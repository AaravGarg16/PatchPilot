from agent.llm import DEFAULT_MODEL, DEFAULT_PROVIDER, LLMSettings, get_chat_model, load_llm_settings


def test_defaults_when_env_unset(monkeypatch):
    monkeypatch.setattr("agent.llm.load_dotenv", lambda: None)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = load_llm_settings()

    assert settings.provider == DEFAULT_PROVIDER
    assert settings.model == DEFAULT_MODEL


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setattr("agent.llm.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5-20251001")

    settings = load_llm_settings()

    assert settings.model == "claude-haiku-4-5-20251001"


def test_builds_chat_model_without_calling_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    model = get_chat_model(LLMSettings(provider="anthropic", model="claude-sonnet-5"))

    assert model.model == "claude-sonnet-5"
