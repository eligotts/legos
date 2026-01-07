"""Server configuration."""

from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Configuration for inference server."""

    model_path: str = "/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-MLX-4bit"
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_tokens: int = 4096

    # Sampler configuration
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    repetition_penalty: float = 1.0  # 1.0 = no penalty, >1.0 penalizes repetition

    # LoRA configuration (sets up empty LoRA layers at startup for weight updates)
    lora_rank: int | None = None  # If set, enables LoRA with this rank
    lora_layers: int = 16  # Number of layers to apply LoRA to (-1 for all)
    lora_scale: float = 32.0  # LoRA scaling factor (lora_alpha)
    lora_keys: list[str] | None = None  # Target specific layer keys (default: attention only)

    model_config = {"env_prefix": "SELF_PLAY_"}
