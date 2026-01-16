"""
Text Reversal: Train proposer and solver with both actors trainable.

The task:
- Proposer generates a string
- Solver must reverse that string character by character
- Both actors are trained (solver episodes are trainable)

Key difference from proposer_solver:
- Solver episodes spawned by proposer are `is_trainable=True`
- Only proposer episodes are scheduled in get_batch()

To run:
1. Configure LoRA parameters in src/legos/lora.py

2. Start inference server:
   legos serve --model /path/to/model

3. Run training:
   uv run examples/train_text_reversal.py
"""
import asyncio
import argparse

import mlx.optimizers as optim
from mlx_lm import load

from legos.core import OpenAIClient, Actor, GRPOCredit
from legos.lora import apply_lora, print_trainable_params
from legos.tasks.text_reversal import (
    TextReversalArena,
    TextReversalProposerEpisode,
    ReverseEpisode,
)
from legos.training import (
    Trainer,
    TrainerConfig,
    training_loop,
    synchronous_training_loop,
)


# =============================================================================
# CONFIGURATION - Edit these values directly
# =============================================================================

# Model
MODEL_PATH = "mlx_model"

# Inference server
INFERENCE_URL = "http://localhost:8000"

# Episode settings
EPISODES_PER_STEP = 4     # Proposer episodes per step
N_SOLVER_ROLLOUTS = 4     # Solver attempts per proposal
TARGET_PASS_RATE = 0.5    # Goldilocks target for proposer reward

# Generation
EPISODE_CONCURRENCY = 4   # Max concurrent episodes
STEP_CONCURRENCY = 1      # Max concurrent arena.step() calls

# Training
NUM_STEPS = 100
LR = 1e-5
MIN_SAMPLES_PER_STEP = 24  # Records needed before optimizer step
MICRO_BATCH_TOKENS = 2048  # Max tokens per micro-batch
STALENESS_LIMIT = 2        # Discard records older than N steps

# KL regularization
KL_COEF = 0.1
USE_KL_PENALTY = False

# Generation parameters
MAX_TOKENS = 256

# Training mode
USE_SIMPLE_LOOP = False  # True for sequential, False for async

# Logging
WANDB_PROJECT = None  # Set to enable W&B logging
WANDB_RUN_NAME = None

# =============================================================================


# System prompts for each actor
PROPOSER_SYSTEM_PROMPT = """You are a string generator. Your task is to create interesting and varied strings of text.

Generate strings that are:
- Between 5 and 50 characters long
- Creative and diverse (avoid repetitive or simple patterns)
- Can include letters, numbers, spaces, and basic punctuation

Always output ONLY the string, with no explanation or additional text."""

SOLVER_SYSTEM_PROMPT = """You are a string reversal specialist. Your task is to reverse strings character by character.

Rules:
- Reverse the entire string from end to beginning
- Preserve all characters including spaces and punctuation
- Output ONLY the reversed string, nothing else

Example:
Input: "hello world"
Output: "dlrow olleh" """


def load_model_with_lora(model_path: str):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    apply_lora(model, inference_mode=False)
    print_trainable_params(model)

    return model, tokenizer


def truncate(text: str, max_len: int = 50) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


async def preview_text_reversal(arena, concurrency: int = 4):
    """Preview text reversal arena output."""
    print("\n=== Text Reversal Preview ===\n")
    batch = await arena.step(concurrency=concurrency)

    if not batch.records:
        print("No records generated.")
        return

    # Group by rollout
    episodes = {}
    for r in batch.records:
        if r.rollout_id not in episodes:
            episodes[r.rollout_id] = {"meta": r.meta, "records": []}
        episodes[r.rollout_id]["records"].append(r)

    # Separate proposer and solver episodes
    proposer_episodes = []
    solver_episodes = []

    for rollout_id, data in episodes.items():
        meta = data["meta"]
        episode_type = meta.get("episode_type", "unknown")
        if episode_type == "text_reversal":
            proposer_episodes.append((rollout_id, data))
        elif episode_type == "reverse":
            solver_episodes.append((rollout_id, data))

    print(f"Proposer episodes: {len(proposer_episodes)}")
    print(f"Solver episodes: {len(solver_episodes)}")
    print(f"Total records: {len(batch.records)}\n")

    # Track stats
    proposer_stats = {"total": 0, "valid": 0}
    solver_stats = {"total": 0, "correct": 0}

    # Show proposer episodes
    print("--- Proposer Episodes ---\n")
    for i, (rollout_id, data) in enumerate(proposer_episodes, 1):
        meta = data["meta"]
        extras = meta.get("extras", {})

        proposed = extras.get("proposed_string", "")
        ground_truth = extras.get("ground_truth", "")
        pass_rate = extras.get("pass_rate", 0.0)
        n_solvers = extras.get("n_solvers", 0)

        proposer_stats["total"] += 1
        if proposed and len(proposed) >= 2:
            proposer_stats["valid"] += 1

        # Get proposer reward
        proposer_reward = None
        for rec in data["records"]:
            if rec.actor_id == "Proposer":
                proposer_reward = rec.reward
                break

        print(f"[{i}] String: \"{truncate(proposed, 40)}\"")
        print(f"    Reversed: \"{truncate(ground_truth, 40)}\"")
        print(f"    Pass rate: {pass_rate:.2f} ({n_solvers} solvers)")
        if proposer_reward is not None:
            print(f"    Proposer reward: {proposer_reward:+.2f}")
        print()

    # Show solver episodes (sample)
    print("--- Solver Episodes (sample) ---\n")
    for i, (rollout_id, data) in enumerate(solver_episodes[:5], 1):
        meta = data["meta"]
        extras = meta.get("extras", {})

        string = extras.get("string", "")
        ground_truth = extras.get("ground_truth", "")
        predicted = extras.get("predicted", "")
        correct = extras.get("correct", False)

        solver_stats["total"] += 1
        if correct:
            solver_stats["correct"] += 1

        # Get solver reward
        solver_reward = None
        for rec in data["records"]:
            solver_reward = rec.reward
            break

        status = "✓" if correct else "✗"
        print(f"[{i}] {status} Input: \"{truncate(string, 30)}\"")
        print(f"    Expected: \"{truncate(ground_truth, 30)}\"")
        print(f"    Got:      \"{truncate(predicted, 30)}\"")
        if solver_reward is not None:
            print(f"    Reward: {solver_reward:+.1f}")
        print()

    # Count remaining solvers
    for rollout_id, data in solver_episodes[5:]:
        meta = data["meta"]
        extras = meta.get("extras", {})
        correct = extras.get("correct", False)
        solver_stats["total"] += 1
        if correct:
            solver_stats["correct"] += 1

    # Summary
    print("--- Summary ---")
    print(f"Proposer: {proposer_stats['valid']}/{proposer_stats['total']} valid strings")
    if solver_stats["total"] > 0:
        acc = solver_stats["correct"] / solver_stats["total"]
        print(f"Solver: {solver_stats['correct']}/{solver_stats['total']} correct ({acc:.1%})")

    all_rewards = [r.reward for r in batch.records]
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        print(f"\nAvg reward: {avg_reward:.3f} | Range: {min(all_rewards):.3f} to {max(all_rewards):.3f}")


async def main(args):
    # Setup inference client
    print(f"\nConnecting to inference server at {INFERENCE_URL}...")
    client = OpenAIClient(
        api_key="not-needed",
        model="local",
        base_url=f"{INFERENCE_URL.rstrip('/')}/v1",
        timeout=120.0,
    )

    # Setup arena
    arena = TextReversalArena(
        client=client,
        episodes_per_step=EPISODES_PER_STEP,
        verbose=args.verbose,
        credit_assigner=GRPOCredit(),
    )

    # Add actors
    arena.add_actor(Actor(
        id="Proposer",
        system_prompt=PROPOSER_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_actor(Actor(
        id="Solver",
        system_prompt=SOLVER_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    # Add episodes
    arena.add_episode("text_reversal", TextReversalProposerEpisode(
        proposer_actor_id="Proposer",
        n_solver_rollouts=N_SOLVER_ROLLOUTS,
        target_pass_rate=TARGET_PASS_RATE,
    ))

    arena.add_episode("reverse", ReverseEpisode(
        solver_actor_id="Solver",
    ))

    # Dry-run mode
    if args.dry_run:
        await preview_text_reversal(arena, concurrency=EPISODE_CONCURRENCY)
        await client.close()
        return

    # Load model with LoRA
    model, tokenizer = load_model_with_lora(MODEL_PATH)

    # Setup trainer
    optimizer = optim.Adam(learning_rate=LR)
    config = TrainerConfig(
        lr=LR,
        micro_batch_tokens=MICRO_BATCH_TOKENS,
        staleness_limit=STALENESS_LIMIT,
        min_samples_per_step=MIN_SAMPLES_PER_STEP,
        ppo_clip_min=0.8,
        ppo_clip_max=1.2,
        kl_coef=KL_COEF,
        use_kl_penalty=USE_KL_PENALTY,
        inference_url=INFERENCE_URL,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
    )

    trainer = Trainer(model=model, optimizer=optimizer, config=config, client=client)

    # Run training
    print(f"\nStarting training for {NUM_STEPS} steps...")
    print(f"  - Task: Text Reversal (Proposer/Solver)")
    print(f"  - Proposer episodes per step: {EPISODES_PER_STEP}")
    print(f"  - Solver rollouts per proposal: {N_SOLVER_ROLLOUTS}")
    print(f"  - Min samples per step: {MIN_SAMPLES_PER_STEP}")
    print()

    try:
        if USE_SIMPLE_LOOP:
            await synchronous_training_loop(
                arena=arena,
                trainer=trainer,
                num_steps=NUM_STEPS,
                episode_concurrency=EPISODE_CONCURRENCY,
                step_concurrency=STEP_CONCURRENCY,
                verbose=args.verbose,
            )
        else:
            batch_queue = asyncio.Queue(maxsize=4)
            await training_loop(
                arena=arena,
                trainer=trainer,
                batch_queue=batch_queue,
                num_steps=NUM_STEPS,
                episode_concurrency=EPISODE_CONCURRENCY,
                step_concurrency=STEP_CONCURRENCY,
                verbose=args.verbose,
            )
    finally:
        await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text reversal agents")

    # Only essential CLI args
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true",
        help="Print debug info")

    args = parser.parse_args()
    asyncio.run(main(args))
