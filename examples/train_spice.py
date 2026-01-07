"""
Example: Training a SPICE (Self-Play In Corpus Environment) agent with RL.

This script demonstrates how to train proposer and solver agents using
corpus-grounded question generation. Inspired by the SPICE paper (arXiv:2510.24684).

Key concepts:
- Proposer reads documents and generates questions
- Solver answers questions without access to source documents
- Proposer is rewarded for questions at the frontier of solver capability (~50% pass rate)
- Solver is rewarded via LLM-as-judge for correctness

To run this example:
1. Start mlx-vllm server with LoRA enabled:
   MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \
   python -m uvicorn mlx_vllm.server:app --port 8000

2. Set your OpenRouter API key (for the LLM judge):
   export OPENROUTER_API_KEY=your-key-here

3. Run this script:
   python examples/train_spice.py --dry-run  # Preview without training
   python examples/train_spice.py            # Full training
"""
import asyncio
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from self_play.core import OpenAIClient, Role, Artifact
from self_play.tasks.spice import SpiceArena, SpiceProposerEpisode, SpiceSolverEpisode
from self_play.training import (
    Trainer,
    TrainerConfig,
    WeightPublisher,
    training_loop,
    simple_training_loop,
)


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

PROPOSER_SYSTEM = """You are a question generator. Given a document, create challenging but answerable questions.
Your questions should:
- Test comprehension and reasoning
- Have clear, unambiguous answers
- Be diverse in style (factual, inferential, analytical)

Always respond with valid JSON only."""

SOLVER_SYSTEM = """You are a knowledgeable question answerer. Answer questions accurately and concisely.
Think step by step when needed, then provide a clear final answer.
Always end your response with: "The answer is: " followed by your answer."""


# Default corpus path (relative to repo root)
DEFAULT_CORPUS_PATH = Path(__file__).parent.parent / "sample_data" / "spice_corpus.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_with_lora(
    model_path: str,
    lora_rank: int = 16,
    lora_layers: int = 16,
    lora_scale: float = 32.0,
    lora_dropout: float = 0.05,
):
    """Load model and attach LoRA adapters."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    # Defaults match official LiquidAI/PEFT recommendations
    lora_keys = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"}

    print(f"Attaching LoRA (rank={lora_rank}, layers={lora_layers}, keys={lora_keys})...")
    lora_config = {
        "rank": lora_rank,
        "scale": lora_scale,
        "dropout": lora_dropout,
        "keys": lora_keys,
    }
    linear_to_lora_layers(model, lora_layers, lora_config)

    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def load_corpus_from_file(filepath: str) -> list:
    """Load corpus from a JSON or JSONL file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    with open(path, "r") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

async def preview_spice(arena, concurrency: int = 4):
    """Preview SPICE arena output with task-specific metrics."""
    print("\n=== SPICE Preview ===\n")
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

    print(f"Episodes: {len(episodes)} | Records: {len(batch.records)}\n")

    # Track stats by episode type
    propose_stats = {"rewards": [], "pass_rates": []}
    solve_stats = {"correct": 0, "total": 0, "rewards": []}

    for i, (rid, data) in enumerate(episodes.items(), 1):
        meta = data["meta"]
        episode_type = meta.get("episode_type", "?")
        extras = meta.get("extras", {})
        artifact = meta.get("artifact", {}) or {}

        if episode_type == "spice_propose":
            # Proposer episode
            proposed = extras.get("proposed_question", {})
            question = proposed.get("question", "N/A") if proposed else "N/A"
            ground_truth = proposed.get("ground_truth", "N/A") if proposed else "N/A"
            pass_rate = extras.get("pass_rate", 0)
            solver_rewards = extras.get("solver_rewards", [])
            reward = data["records"][0].reward if data["records"] else 0

            propose_stats["rewards"].append(reward)
            propose_stats["pass_rates"].append(pass_rate)

            # Get source document info
            source_doc = extras.get("source_document", {})
            doc_title = source_doc.get("title", "Untitled") if isinstance(source_doc, dict) else "N/A"

            print(f"[{i}] PROPOSE | reward={reward:.2f} | pass_rate={pass_rate:.1%}")
            print(f"    Source: {doc_title}")
            print(f"    Q: {truncate(question, 80)}")
            print(f"    A: {ground_truth}")
            if solver_rewards:
                correct = sum(1 for r in solver_rewards if r > 0.5)
                print(f"    Solver results: {correct}/{len(solver_rewards)} correct")

        elif episode_type == "spice_solve":
            # Solver episode
            question = artifact.get("question", "N/A")
            ground_truth = artifact.get("ground_truth", "N/A")
            reward = data["records"][0].reward if data["records"] else 0
            is_correct = reward > 0.5

            solve_stats["total"] += 1
            if is_correct:
                solve_stats["correct"] += 1
            solve_stats["rewards"].append(reward)

            status = "CORRECT" if is_correct else "WRONG"
            print(f"[{i}] SOLVE | {status} | reward={reward:.2f}")
            print(f"    Q: {truncate(question, 80)}")
            print(f"    Expected: {ground_truth}")

        print()

    # Summary stats
    print("=" * 50)
    print("Summary:")
    if propose_stats["rewards"]:
        avg_reward = sum(propose_stats["rewards"]) / len(propose_stats["rewards"])
        avg_pass_rate = sum(propose_stats["pass_rates"]) / len(propose_stats["pass_rates"])
        print(f"  Proposer: {len(propose_stats['rewards'])} episodes, "
              f"avg_reward={avg_reward:.3f}, avg_pass_rate={avg_pass_rate:.1%}")

    if solve_stats["total"] > 0:
        accuracy = solve_stats["correct"] / solve_stats["total"]
        avg_reward = sum(solve_stats["rewards"]) / len(solve_stats["rewards"])
        print(f"  Solver: {solve_stats['correct']}/{solve_stats['total']} correct ({accuracy:.1%}), "
              f"avg_reward={avg_reward:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args):
    # Load corpus
    corpus_path = args.corpus_file or DEFAULT_CORPUS_PATH
    print(f"Loading corpus from {corpus_path}...")
    corpus = load_corpus_from_file(str(corpus_path))

    print(f"Corpus size: {len(corpus)} documents\n")

    # Setup inference server URL
    base_url = args.url if args.url else f"http://{args.host}:{args.port}"
    print(f"Connecting to inference server at {base_url}...")

    # Setup inference client
    if args.url:
        client = OpenAIClient(
            api_key="not-needed",
            model="local",
            base_url=f"{base_url.rstrip('/')}/v1",
            timeout=120.0,
        )
    else:
        client = OpenAIClient.for_local(host=args.host, port=args.port, timeout=120.0)

    # Setup arena
    print("Setting up SPICE arena...")
    arena = SpiceArena(client=client, batch_size=args.batch_size, verbose=args.verbose)

    # Add roles
    arena.add_role(Role(
        id="Proposer",
        system_prompt=PROPOSER_SYSTEM,
        temperature=0.9,
        max_tokens=512,
    ))

    arena.add_role(Role(
        id="Solver",
        system_prompt=SOLVER_SYSTEM,
        temperature=0.7,
        max_tokens=1024,
    ))

    # Add episodes
    arena.add_episode("spice_propose", SpiceProposerEpisode(
        n_solver_rollouts=args.n_solver_rollouts,
        target_pass_rate=args.target_pass_rate,
    ))
    arena.add_episode("spice_solve", SpiceSolverEpisode())

    # Add stores
    corpus_store = arena.add_store("corpus")
    questions_store = arena.add_store("questions")

    # Populate corpus
    for doc in corpus:
        doc_id = doc.get("id", f"doc_{corpus_store.count()}")
        corpus_store.add(Artifact(id=doc_id, data=doc))

    print(f"Loaded {corpus_store.count()} documents into corpus store")

    # Dry-run mode: preview arena performance without training
    if args.dry_run:
        await preview_spice(arena, concurrency=args.concurrency)
        await client.close()
        return

    # Load model with LoRA
    model, tokenizer = load_model_with_lora(
        model_path=args.model_path,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
    )

    # Setup optimizer
    optimizer = optim.Adam(learning_rate=args.lr)

    # Setup trainer config
    config = TrainerConfig(
        lr=args.lr,
        micro_token_budget=args.micro_token_budget,
        max_policy_lag=args.max_policy_lag,
        batch_size=args.train_batch_size,
        clip_low=0.8,
        clip_high=1.2,
        kl_coef=args.kl_coef,
        use_kl_penalty=args.use_kl_penalty,
        weight_push_url=base_url,
        pad_token_id=tokenizer.pad_token_id or 0,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Setup weight publisher
    publisher = WeightPublisher(base_url=base_url)

    # Setup trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        publisher=publisher,
    )

    print(f"\nStarting SPICE training for {args.num_steps} steps...")
    print(f"  - Corpus size: {corpus_store.count()} documents")
    print(f"  - Batch size: {args.batch_size} proposer episodes per arena step")
    print(f"  - Solver rollouts per proposal: {args.n_solver_rollouts}")
    print(f"  - Target pass rate: {args.target_pass_rate:.0%}")
    print(f"  - Train batch size: {args.train_batch_size} records per train step")
    print(f"  - Micro token budget: {args.micro_token_budget}")
    print()

    if args.simple_loop:
        await simple_training_loop(
            arena=arena,
            trainer=trainer,
            num_steps=args.num_steps,
            concurrency=args.concurrency,
            step_concurrency=args.step_concurrency,
            verbose=args.verbose,
        )
    else:
        batch_queue = asyncio.Queue(maxsize=4)
        await training_loop(
            arena=arena,
            trainer=trainer,
            batch_queue=batch_queue,
            num_steps=args.num_steps,
            concurrency=args.concurrency,
            step_concurrency=args.step_concurrency,
            verbose=args.verbose,
        )

    # Cleanup
    await publisher.close()
    await client.close()

    print(f"\nTraining complete! Final step: {trainer.train_step_idx}")
    print(f"Questions generated: {questions_store.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SPICE (Self-Play In Corpus Environment) agents with RL"
    )

    # Model args
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit",
        help="Path to the base model",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=16, help="LoRA layers")

    # Server args
    parser.add_argument("--url", type=str, default=None, help="Full base URL of inference server")
    parser.add_argument("--host", type=str, default="localhost", help="Inference server host")
    parser.add_argument("--port", type=int, default=8000, help="Inference server port")

    # Corpus args
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help=f"Path to corpus file (JSON or JSONL). Default: sample_data/spice_corpus.json",
    )

    # Training args
    parser.add_argument("--num-steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Proposer episodes per arena step")
    parser.add_argument("--train-batch-size", type=int, default=24, help="Records per train step")
    parser.add_argument("--n-solver-rollouts", type=int, default=4, help="Solver rollouts per proposal")
    parser.add_argument("--target-pass-rate", type=float, default=0.5, help="Target solver pass rate")
    parser.add_argument("--micro-token-budget", type=int, default=4096, help="Tokens per micro-batch")
    parser.add_argument("--max-policy-lag", type=int, default=3, help="Max staleness (steps)")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--use-kl-penalty", action="store_true", help="Add KL penalty to loss")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent episodes")
    parser.add_argument("--step-concurrency", type=int, default=1, help="Max concurrent arena.step() calls")

    # Mode args
    parser.add_argument("--simple-loop", action="store_true", help="Use simple sequential loop")
    parser.add_argument("--dry-run", action="store_true", help="Preview without training")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")

    # Wandb args
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()
    asyncio.run(main(args))
