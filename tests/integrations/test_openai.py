from src.integrations.openai import Settings

def test_settings():
    settings = Settings()
    assert settings.PROJECT_NAME == "llm-playground"
    assert settings.API_VERSION == "v1"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.MODEL_NAME == "gpt-3.5-turbo"
    assert settings.TEMPERATURE == 0.3
    assert settings.MAX_TOKENS == 2048
