"""
Training configuration for the RL trainer.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TrainerConfig:
    """
    Configuration for the RL trainer.

    This trainer is designed as a clean, minimal implementation of core RL
    training patterns for MLX.
    """

    # === Optimizer ===
    lr: float = 1e-5

    # === Batch Sizing ===
    # Minimum training records before optimizer step triggers
    min_samples_per_step: int = 32

    # Max tokens per micro-batch (for gradient accumulation)
    micro_batch_tokens: int = 4096

    # === Off-Policy Handling ===
    # Discard records from policies more than N steps behind current
    staleness_limit: int = 3

    # === PPO Clipping ===
    # Importance ratio clipping bounds [min, max]
    ppo_clip_min: float = 0.8
    ppo_clip_max: float = 1.2

    # Skip batch if clip fraction exceeds this threshold
    clip_skip_threshold: float = 0.3

    # === Loss Configuration ===
    # Loss normalization: "token" (DAPO) or "sample" (GRPO)
    loss_type: Literal["token", "sample"] = "token"

    # Importance sampling level: "token" or "sequence" (GSPO)
    importance_sampling: Literal["token", "sequence"] = "token"

    # GSPO clip epsilon (only used when importance_sampling="sequence")
    gspo_clip_epsilon: float = 3e-4

    # === KL Regularization ===
    # KL coefficient for advantage shaping (subtracts kl_coef * KL from advantages)
    kl_coef: float = 0.1

    # Whether to enable KL advantage shaping
    use_kl_penalty: bool = False

    # Maximum KL value before clipping
    kl_max: float = 0.5

    # === Infrastructure ===
    # URL of mlx-vllm inference server for weight publishing
    inference_url: str = "http://localhost:8000"

    # Pad token ID for collation
    pad_token_id: int = 0

    # Memory management: eval after each micro-batch to free memory
    eval_per_micro_batch: bool = True

    # === Logging ===
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # === Evaluation ===
    eval_every: int = 10  # Run eval every N steps (0 = disabled)
    eval_concurrency: int = 8

    # === Checkpointing ===
    checkpoint_every: int = 0  # Save every N steps (0 = disabled)
    checkpoint_dir: Optional[str] = None

    # === Memory Optimization ===
    # Chunk size for fused LM head computation (memory-efficient logprobs)
    # None = auto (enabled for vocab >= 32K with chunk_size=2048)
    # 0 = disabled (use full materialization)
    # int > 0 = explicit chunk size
    fused_lm_head_chunk_size: Optional[int] = None

    # Gradient checkpointing: recompute activations during backward pass
    # Trades ~33% more compute for significantly less peak memory
    gradient_checkpointing: bool = False
