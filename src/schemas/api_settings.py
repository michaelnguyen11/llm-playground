from abc import abstractmethod, ABC
from typing import Optional
from pydantic import BaseModel, validator


# pylint: disable=too-many-instance-attributes
class AbstractAPISettings(BaseModel, ABC):
    type: Optional[str] = None
    key: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None  # if type is llamacpp, engine is required
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    n_gpu_layers: Optional[int] = None  # determines how many layers of the model are offloaded to your GPU
    n_batch: Optional[int] = None  # how many tokens are processed in parallel
    f16_kv: Optional[bool] = None  # for some reason, Metal only support True

    @classmethod
    @abstractmethod
    def from_defaults(cls):
        """Return a new instance of this class with default values filled in."""
        pass

    @validator("type")
    def validate_type(cls, v):
        if v not in ["openai", "llamacpp"]:
            raise ValueError("type must be openai or llamacpp")
        return v

    @validator("engine")
    def validate_engine(cls, v, values):
        if values.get("type") == "llamacpp" and v not in ["cpu", "gpu", "metal"]:
            raise ValueError("with llamacpp type, engine must be cpu, gpu, or metal")
        return v

    @validator("key")
    def validate_key(cls, v, values):
        if values.get("type") == "openai" and v is None:
            raise ValueError("key is required for openai type")
        return v
