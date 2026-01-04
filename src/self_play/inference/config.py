"""Server configuration."""

from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Configuration for inference server."""

    model_path: str = "/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-MLX-4bit"
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_tokens: int = 4096

    # LoRA configuration (sets up empty LoRA layers at startup for weight updates)
    lora_rank: int | None = None  # If set, enables LoRA with this rank
    lora_layers: int = 16  # Number of layers to apply LoRA to (-1 for all)
    lora_scale: float = 20.0  # LoRA scaling factor

    model_config = {"env_prefix": "SELF_PLAY_"}
