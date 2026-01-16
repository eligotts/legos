"""
GAN: Train Generator vs Discriminator adversarial agents.

The task follows the proposer_solver pattern:
- Generator creates fake content, spawns Monte Carlo discriminator episodes
- Discriminator classifies content as REAL or FAKE
- Generator reward = 1 - pass_rate (wants to fool discriminators)
- Discriminator episodes are trainable when spawned by generator

Arena schedules both:
- GeneratorEpisodes (fake content, spawn disc sub-episodes)
- DiscriminatorEpisodes (real content from store)

Output format: Both actors use \\boxed{} for final answer.

To run:
1. Configure LoRA parameters in src/legos/lora.py

2. Start inference server:
   legos serve --model /path/to/model

3. Run training:
   uv run examples/train_gan.py
"""
import asyncio
import argparse
from typing import Dict, List

import mlx.optimizers as optim
from mlx_lm import load

from legos.core import OpenAIClient, Actor, Artifact, RAECredit
from legos.lora import apply_lora, print_trainable_params
from legos.tasks.gan import (
    GANArena,
    GeneratorEpisode,
    DiscriminatorEpisode,
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
GENERATOR_EPISODES_PER_STEP = 4      # Generator episodes per step
DISCRIMINATOR_EPISODES_PER_STEP = 4  # Standalone discriminator episodes (real data)
N_DISCRIMINATOR_ROLLOUTS = 4         # Discriminator attempts per generated content
FEW_SHOT_K = 3                       # Examples shown to generator

# Generation
EPISODE_CONCURRENCY = 4  # Max concurrent episodes
STEP_CONCURRENCY = 1     # Max concurrent arena.step() calls

# Training
NUM_STEPS = 100
LR = 1e-5
MIN_SAMPLES_PER_STEP = 24  # Records needed before optimizer step
MICRO_BATCH_TOKENS = 2048  # Max tokens per micro-batch
STALENESS_LIMIT = 2        # Discard records older than N steps

# KL regularization
KL_COEF = 0.1
USE_KL_PENALTY = False

# RAE credit assignment
RAE_DECAY = 0.95

# Generation parameters
MAX_TOKENS = 512

# Training mode
USE_SIMPLE_LOOP = False  # True for sequential, False for async

# Logging
WANDB_PROJECT = None  # Set to enable W&B logging
WANDB_RUN_NAME = None

# =============================================================================


# System prompts for each actor
GENERATOR_SYSTEM_PROMPT = """You are a creative content generator. Your task is to create realistic content that is indistinguishable from authentic examples.

When generating content:
1. Think carefully about what makes the examples feel authentic
2. Match the style, tone, and quality
3. Create something that could plausibly be real

Always think through your approach, then provide your final generated content in a \\boxed{} command."""

DISCRIMINATOR_SYSTEM_PROMPT = """You are a discriminator tasked with identifying AI-generated content. Analyze content carefully and determine if it is REAL (authentic human-created) or FAKE (AI-generated).

Look for telltale signs:
- Unusual phrasing or overly generic language
- Lack of specific details that real content would have
- Patterns typical of AI generation
- Inconsistencies or logical errors

Always think through your analysis, then provide your final judgment in \\boxed{REAL} or \\boxed{FAKE}."""


# =============================================================================
# SAMPLE DATA - Replace with your real data
# =============================================================================

SAMPLE_REAL_DATA: List[Dict[str, str]] = [
    {"content": "The morning light filtered through the café window as Sarah stirred her coffee. Three years since she'd been back in this town, and everything felt both familiar and foreign. The owner, Mr. Chen, still remembered her order—black coffee, no sugar, side of optimism."},
    {"content": "Meeting notes from Q4 planning: Revenue targets exceeded by 12%. Customer retention at 94%. Key initiatives for next quarter include mobile app launch (Feb 15), APAC expansion (March), and platform upgrade to v3.0. Action items assigned to respective team leads."},
    {"content": "Recipe: Grandma's Apple Pie\n- 6 Granny Smith apples, peeled and sliced\n- 3/4 cup sugar\n- 2 tbsp flour\n- 1 tsp cinnamon\n- 1/4 tsp nutmeg\n- Pre-bake crust at 425°F for 15 min. Fill and bake covered for 45 min, then uncovered for 15 min until golden."},
    {"content": "The experiment results were unexpected. When we increased the temperature to 450K, the catalyst showed a 40% improvement in conversion rate. Dr. Martinez suggested this might be due to the phase transition we observed at 420K. Further trials are scheduled for next week."},
    {"content": "Lost: Orange tabby cat, male, neutered. Answers to 'Mango'. Last seen near Oak Street Park on Tuesday evening. He has a small notch in his left ear and wears a blue collar with tags. Reward offered. Please call if found."},
    {"content": "The interview went well, I think. They asked about my experience with distributed systems and I talked about the caching layer I built at my last job. The team seemed friendly. They mentioned they'd get back to me within two weeks."},
    {"content": "Customer review: 4/5 stars. The product arrived on time and works as described. Assembly took longer than expected (about 2 hours), but the instructions were clear. Only complaint is the color is slightly darker than shown in photos. Would recommend."},
    {"content": "Trail conditions for Mt. Wilson: Moderate difficulty, 14.2 miles round trip, 4,600 ft elevation gain. Snow above 8,000 ft, crampons recommended. Trailhead parking lot was full by 7am on Saturday. Bring extra water—all streams were dry."},
]


def load_real_examples() -> List[Dict[str, str]]:
    """
    Load real examples for training.

    Replace this function with your actual data loading logic.
    Data should be a list of dicts with at least a 'content' field.
    """
    return SAMPLE_REAL_DATA


def load_model_with_lora(model_path: str):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    apply_lora(model, inference_mode=False)
    print_trainable_params(model)

    return model, tokenizer


def truncate(text: str, max_len: int = 100) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


async def preview_gan(arena, concurrency: int = 4):
    """Preview GAN arena output with metrics."""
    print("\n=== GAN Preview ===\n")
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

    # Separate generator and discriminator episodes
    generator_episodes = []
    discriminator_episodes = []

    for rollout_id, data in episodes.items():
        meta = data["meta"]
        episode_type = meta.get("episode_type", "unknown")
        if episode_type == "generate":
            generator_episodes.append((rollout_id, data))
        elif episode_type == "discriminate":
            discriminator_episodes.append((rollout_id, data))

    print(f"Generator episodes: {len(generator_episodes)}")
    print(f"Discriminator episodes: {len(discriminator_episodes)}")
    print(f"Total records: {len(batch.records)}\n")

    # Track stats
    gen_stats = {"total": 0, "valid": 0, "avg_pass_rate": []}
    disc_stats = {"real_correct": 0, "real_wrong": 0, "fake_correct": 0, "fake_wrong": 0, "invalid": 0}

    # Show generator episodes
    print("--- Generator Episodes ---\n")
    for i, (rollout_id, data) in enumerate(generator_episodes, 1):
        meta = data["meta"]
        extras = meta.get("extras", {})

        generated = extras.get("generated_content", "")
        all_judgments = extras.get("all_judgments", [])
        n_valid = extras.get("n_valid_judgments", 0)
        pass_rate = extras.get("pass_rate", 0.0)

        gen_stats["total"] += 1
        if generated and len(generated.strip()) >= 2:
            gen_stats["valid"] += 1
        if n_valid > 0:
            gen_stats["avg_pass_rate"].append(pass_rate)

        # Get generator reward
        gen_reward = None
        for rec in data["records"]:
            if rec.actor_id == "Generator":
                gen_reward = rec.reward
                break

        print(f"[{i}] Generated: \"{truncate(generated, 60)}\"")
        print(f"    Judgments: {all_judgments} (valid: {n_valid})")
        print(f"    Disc pass rate: {pass_rate:.2f}")
        if gen_reward is not None:
            print(f"    Generator reward: {gen_reward:+.2f} (1 - pass_rate)")
        print()

    # Show discriminator episodes (sample)
    print("--- Discriminator Episodes (standalone on real data) ---\n")
    for i, (rollout_id, data) in enumerate(discriminator_episodes[:5], 1):
        meta = data["meta"]
        extras = meta.get("extras", {})

        content = extras.get("content", "")
        is_real = extras.get("is_real", True)
        judgment = extras.get("discriminator_judgment", "invalid")
        boxed = extras.get("boxed_output", "")

        # Track stats
        if judgment == "invalid":
            disc_stats["invalid"] += 1
        elif is_real:
            if judgment == "real":
                disc_stats["real_correct"] += 1
            else:
                disc_stats["real_wrong"] += 1
        else:
            if judgment == "fake":
                disc_stats["fake_correct"] += 1
            else:
                disc_stats["fake_wrong"] += 1

        # Get discriminator reward
        disc_reward = None
        for rec in data["records"]:
            disc_reward = rec.reward
            break

        label = "REAL" if is_real else "FAKE"
        correct = (is_real and judgment == "real") or (not is_real and judgment == "fake")
        status = "✓" if correct else "✗" if judgment != "invalid" else "?"

        print(f"[{i}] {status} Content ({label}): \"{truncate(content, 50)}\"")
        print(f"    Judgment: {judgment} (boxed: {boxed})")
        if disc_reward is not None:
            print(f"    Reward: {disc_reward:+.1f}")
        print()

    # Count remaining discriminators
    for rollout_id, data in discriminator_episodes[5:]:
        meta = data["meta"]
        extras = meta.get("extras", {})
        is_real = extras.get("is_real", True)
        judgment = extras.get("discriminator_judgment", "invalid")

        if judgment == "invalid":
            disc_stats["invalid"] += 1
        elif is_real:
            if judgment == "real":
                disc_stats["real_correct"] += 1
            else:
                disc_stats["real_wrong"] += 1
        else:
            if judgment == "fake":
                disc_stats["fake_correct"] += 1
            else:
                disc_stats["fake_wrong"] += 1

    # Summary
    print("--- Summary ---")
    print(f"Generator: {gen_stats['valid']}/{gen_stats['total']} valid generations")
    if gen_stats["avg_pass_rate"]:
        avg_pr = sum(gen_stats["avg_pass_rate"]) / len(gen_stats["avg_pass_rate"])
        print(f"  Avg disc pass rate: {avg_pr:.2f} (gen wants this low)")

    total_disc = disc_stats["real_correct"] + disc_stats["real_wrong"] + disc_stats["fake_correct"] + disc_stats["fake_wrong"]
    if total_disc > 0:
        print(f"Discriminator (standalone on real):")
        print(f"  Real: {disc_stats['real_correct']} correct, {disc_stats['real_wrong']} wrong")
        print(f"  Invalid: {disc_stats['invalid']}")

    all_rewards = [r.reward for r in batch.records]
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        print(f"\nAvg reward: {avg_reward:.3f} | Range: {min(all_rewards):.3f} to {max(all_rewards):.3f}")


async def main(args):
    # Load real examples
    print("Loading real examples...")
    real_examples = load_real_examples()
    print(f"Loaded {len(real_examples)} real examples")

    # Setup inference client
    print(f"\nConnecting to inference server at {INFERENCE_URL}...")
    client = OpenAIClient(
        api_key="not-needed",
        model="local",
        base_url=f"{INFERENCE_URL.rstrip('/')}/v1",
        timeout=120.0,
    )

    # Setup arena
    arena = GANArena(
        client=client,
        generator_episodes_per_step=GENERATOR_EPISODES_PER_STEP,
        discriminator_episodes_per_step=DISCRIMINATOR_EPISODES_PER_STEP,
        n_discriminator_rollouts=N_DISCRIMINATOR_ROLLOUTS,
        few_shot_k=FEW_SHOT_K,
        verbose=args.verbose,
        credit_assigner=RAECredit(decay=RAE_DECAY),
    )

    # Add actors
    arena.add_actor(Actor(
        id="Generator",
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    arena.add_actor(Actor(
        id="Discriminator",
        system_prompt=DISCRIMINATOR_SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
    ))

    # Add episodes
    arena.add_episode("generate", GeneratorEpisode(
        generator_actor_id="Generator",
        n_discriminator_rollouts=N_DISCRIMINATOR_ROLLOUTS,
    ))

    arena.add_episode("discriminate", DiscriminatorEpisode(
        discriminator_actor_id="Discriminator",
    ))

    # Load real examples into store
    store = arena.add_store("real_examples")
    for i, item in enumerate(real_examples):
        store.add(Artifact(id=f"real_{i}", data=item))
    print(f"Added {store.count()} examples to store")

    # Dry-run mode
    if args.dry_run:
        await preview_gan(arena, concurrency=EPISODE_CONCURRENCY)
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
    print(f"  - Task: GAN (Generator vs Discriminator)")
    print(f"  - Generator episodes per step: {GENERATOR_EPISODES_PER_STEP}")
    print(f"  - Discriminator episodes per step: {DISCRIMINATOR_EPISODES_PER_STEP}")
    print(f"  - Discriminator rollouts per generation: {N_DISCRIMINATOR_ROLLOUTS}")
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
    parser = argparse.ArgumentParser(description="Train GAN adversarial agents")

    # Only essential CLI args
    parser.add_argument("--dry-run", action="store_true",
        help="Run single arena step to preview performance (skips training)")
    parser.add_argument("--verbose", action="store_true",
        help="Print debug info")

    args = parser.parse_args()
    asyncio.run(main(args))
