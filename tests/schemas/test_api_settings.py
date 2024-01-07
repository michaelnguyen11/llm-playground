import pytest
from pydantic import ValidationError
from src.schemas.api_settings import AbstractAPISettings


class MockAbstractAPISettings(AbstractAPISettings):
    @classmethod
    def from_defaults(cls):
        pass

def test_validate_type():
    with pytest.raises(ValidationError):
        MockAbstractAPISettings(type="invalid_type")

    try:
        MockAbstractAPISettings(type="openai")
        MockAbstractAPISettings(type="llamacpp")
    except ValidationError:
        pytest.fail("ValidationError was raised unexpectedly!")


def test_validate_engine():
    with pytest.raises(ValidationError):
        MockAbstractAPISettings(type="llamacpp", engine="invalid_engine")

    try:
        MockAbstractAPISettings(type="llamacpp", engine="cpu")
        MockAbstractAPISettings(type="llamacpp", engine="gpu")
        MockAbstractAPISettings(type="llamacpp", engine="metal")
    except ValidationError:
        pytest.fail("ValidationError was raised unexpectedly!")


def test_validate_key():
    with pytest.raises(ValidationError):
        MockAbstractAPISettings(type="openai", key=None)

    try:
        MockAbstractAPISettings(type="openai", key="valid_key")
    except ValidationError:
        pytest.fail("ValidationError was raised unexpectedly!")
