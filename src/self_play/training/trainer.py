"""
Trainer class for RL training with gradient accumulation.

The Trainer orchestrates:
1. Filtering stale records based on policy version
2. Splitting into micro-batches by token budget
3. Gradient accumulation across micro-batches
4. Optimizer step
5. Weight publishing to inference server
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from ..core.arena import TrainingBatch
from ..core.types import TrainingRecord
from .config import TrainerConfig
from .batching import split_by_token_budget, collate
from .loss import make_loss_fn
from .weight_publisher import WeightPublisher


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str, timings: dict):
        self.name = name
        self.timings = timings

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        if self.name not in self.timings:
            self.timings[self.name] = []
        self.timings[self.name].append(elapsed)


class Trainer:
    """
    RL trainer with gradient accumulation and weight publishing.

    This is a clean, minimal implementation that embodies the core patterns
    of modern RL training:
    - Staleness filtering by policy version
    - Token-budget-based micro-batching
    - Gradient accumulation
    - Online weight updates to inference server

    Example:
        model, tokenizer = load_model_with_lora(...)
        optimizer = mx.optimizers.Adam(learning_rate=1e-5)
        config = TrainerConfig(use_importance_sampling=False)
        publisher = WeightPublisher(base_url="http://localhost:8000")

        trainer = Trainer(model, optimizer, config, publisher)
        metrics = await trainer.train_step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        config: TrainerConfig,
        publisher: Optional[WeightPublisher] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: MLX model with LoRA attached (trainable parameters)
            optimizer: MLX optimizer (e.g., mx.optimizers.Adam)
            config: TrainerConfig with hyperparameters
            publisher: Optional WeightPublisher for hot-swap to inference server
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.publisher = publisher

        self.train_step_idx = 0

        # Create loss function for value_and_grad
        self._loss_fn = make_loss_fn(
            use_importance_sampling=config.use_importance_sampling,
            clip_low=config.clip_low,
            clip_high=config.clip_high,
        )

    def filter_stale_records(
        self,
        records: List[TrainingRecord],
    ) -> List[TrainingRecord]:
        """
        Filter out records that are too old based on policy version.

        Args:
            records: List of training records

        Returns:
            List of fresh records (within max_policy_lag steps of current)
        """
        fresh = []
        for r in records:
            policy_version = r.meta.get("policy_version", 0)
            lag = self.train_step_idx - policy_version
            if lag <= self.config.max_policy_lag:
                fresh.append(r)
        return fresh

    def _train_step_sync(
        self,
        fresh_records: List[TrainingRecord],
        verbose: bool = False,
    ) -> Tuple[Dict[str, float], int]:
        """
        Synchronous training step - runs in a separate thread.

        This contains all the CPU/GPU-bound MLX work that would otherwise
        block the asyncio event loop.

        Args:
            fresh_records: Pre-filtered list of fresh training records
            verbose: Print debug info

        Returns:
            Tuple of (metrics dict, new train_step_idx)
        """
        step_start = time.perf_counter()
        timings: Dict[str, list] = {}

        # Set memory limit (like GRPO trainer does) - helps with memory pressure
        try:
            mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
        except Exception:
            pass  # May not be available on all systems

        # Split into micro-batches by token budget
        with Timer("split_batches", timings):
            micro_batches = split_by_token_budget(
                fresh_records,
                self.config.micro_token_budget,
            )

        if verbose:
            total_tokens_preview = sum(len(r.completion_token_ids) for r in fresh_records)
            total_input_tokens = sum(len(r.input_ids) for r in fresh_records)
            print(f"  → {len(fresh_records)} fresh records, {total_tokens_preview} completion tokens, "
                  f"{total_input_tokens} total input tokens → {len(micro_batches)} micro-batches")

        # Gradient accumulation loop
        accumulated_grads = None
        total_tokens = 0
        total_records = len(fresh_records)

        # Track lazy losses/metrics for batched eval at the end
        micro_losses = []  # List of (loss, num_records)
        micro_metrics = []  # List of (metrics_dict, num_records)

        # Create value_and_grad function
        loss_and_grad_fn = nn.value_and_grad(self.model, self._loss_fn)

        for micro_idx, micro in enumerate(micro_batches):
            micro_start = time.perf_counter()
            micro_timings: Dict[str, float] = {}

            # Collate into tensors
            t0 = time.perf_counter()
            input_ids, loss_mask, inference_logprobs, advantages = collate(
                micro,
                pad_token_id=self.config.pad_token_id,
            )
            micro_timings["collate"] = time.perf_counter() - t0

            # Log tensor shapes for debugging
            if verbose:
                seq_lengths = [len(r.input_ids) for r in micro]
                print(f"    [micro {micro_idx + 1}] input_ids shape: {input_ids.shape}, "
                      f"records: {len(micro)}, "
                      f"seq_lens: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.0f}")

            # Compute loss, metrics, and gradients in a SINGLE forward pass
            # nn.value_and_grad diffs w.r.t. first return value (loss), passes through rest (metrics)
            # Returns ((loss, metrics), grads)
            t0 = time.perf_counter()
            (loss, metrics), grads = loss_and_grad_fn(
                self.model,
                input_ids,
                inference_logprobs,
                advantages,
                loss_mask,
            )
            micro_timings["loss_and_grad_lazy"] = time.perf_counter() - t0

            # For memory-constrained systems: eval immediately to free activations
            if self.config.eval_per_micro_batch:
                t0 = time.perf_counter()
                mx.eval(loss, metrics["clip_fraction"], metrics["kl"], grads)
                micro_timings["eval"] = time.perf_counter() - t0

            # Accumulate gradients weighted by record proportion
            t0 = time.perf_counter()
            weight = len(micro) / total_records
            if accumulated_grads is None:
                accumulated_grads = tree_map(lambda g: g * weight, grads)
            else:
                accumulated_grads = tree_map(
                    lambda a, g: a + g * weight,
                    accumulated_grads,
                    grads,
                )
            micro_timings["grad_accumulate"] = time.perf_counter() - t0

            # Track metrics
            micro_tokens = sum(len(r.completion_token_ids) for r in micro)
            micro_losses.append((loss, len(micro)))
            micro_metrics.append((metrics, len(micro)))
            total_tokens += micro_tokens

            # Clear cache after each micro-batch to free memory
            if self.config.eval_per_micro_batch:
                t0 = time.perf_counter()
                mx.clear_cache()
                micro_timings["clear_cache"] = time.perf_counter() - t0

            micro_total = time.perf_counter() - micro_start

            if verbose:
                micro_loss = float(loss.item()) if self.config.eval_per_micro_batch else 0
                micro_kl = float(metrics["kl"].item()) if self.config.eval_per_micro_batch else 0
                micro_clip = float(metrics["clip_fraction"].item()) if self.config.eval_per_micro_batch else 0
                print(f"    micro-batch {micro_idx + 1}/{len(micro_batches)}: "
                      f"{len(micro)} records, {micro_tokens} completion_tokens, "
                      f"loss={micro_loss:.4f}, kl={micro_kl:.4f}, clip={micro_clip:.2%}")
                if self.config.eval_per_micro_batch:
                    print(f"      ⏱️  collate={micro_timings['collate']*1000:.1f}ms, "
                          f"fwd+bwd(lazy)={micro_timings['loss_and_grad_lazy']*1000:.1f}ms, "
                          f"eval={micro_timings['eval']*1000:.1f}ms, "
                          f"accum={micro_timings['grad_accumulate']*1000:.1f}ms, "
                          f"clear={micro_timings['clear_cache']*1000:.1f}ms, "
                          f"TOTAL={micro_total*1000:.1f}ms")
                else:
                    print(f"      ⏱️  collate={micro_timings['collate']*1000:.1f}ms, "
                          f"fwd+bwd(lazy)={micro_timings['loss_and_grad_lazy']*1000:.1f}ms, "
                          f"accum(lazy)={micro_timings['grad_accumulate']*1000:.1f}ms, "
                          f"TOTAL={micro_total*1000:.1f}ms")

        # Optimizer step
        t0 = time.perf_counter()
        self.optimizer.update(self.model, accumulated_grads)
        optimizer_update_time = time.perf_counter() - t0

        if verbose:
            print(f"  → optimizer update: {optimizer_update_time*1000:.1f}ms")

        # Evaluate model parameters (and losses/metrics if not done per-micro-batch)
        t0 = time.perf_counter()
        if self.config.eval_per_micro_batch:
            # Already evaluated losses/metrics per micro-batch, just need to eval model params
            mx.eval(self.model.parameters())
        else:
            # Batch eval: model params + all losses + all metrics
            all_losses = [loss for loss, _ in micro_losses]
            all_clip_fracs = [m["clip_fraction"] for m, _ in micro_metrics]
            all_kls = [m["kl"] for m, _ in micro_metrics]
            mx.eval(self.model.parameters(), *all_losses, *all_clip_fracs, *all_kls)
        eval_time = time.perf_counter() - t0

        if verbose:
            try:
                peak_mem_gb = mx.get_peak_memory() / 1e9
                active_mem_gb = mx.get_active_memory() / 1e9
                print(f"  → final eval (params): {eval_time*1000:.1f}ms ({eval_time:.2f}s)")
                print(f"  → memory: peak={peak_mem_gb:.2f}GB, active={active_mem_gb:.2f}GB")
            except Exception:
                print(f"  → final eval (params): {eval_time*1000:.1f}ms ({eval_time:.2f}s)")

        # Extract scalar values (arrays should already be materialized if eval_per_micro_batch)
        total_loss = sum(float(loss.item()) * n for loss, n in micro_losses)
        total_clip_fraction = sum(float(m["clip_fraction"].item()) * n for m, n in micro_metrics)
        total_kl = sum(float(m["kl"].item()) * n for m, n in micro_metrics)

        # Clear cache
        t0 = time.perf_counter()
        mx.clear_cache()
        clear_cache_time = time.perf_counter() - t0

        if verbose:
            print(f"  → clear_cache: {clear_cache_time*1000:.1f}ms")

        # Increment step counter
        self.train_step_idx += 1

        step_total = time.perf_counter() - step_start

        # Compute averages
        num_fresh = len(fresh_records)
        avg_loss = total_loss / num_fresh
        avg_kl = total_kl / num_fresh
        avg_clip = total_clip_fraction / num_fresh

        if verbose:
            print(f"  → step complete: avg_loss={avg_loss:.4f}, avg_kl={avg_kl:.4f}, "
                  f"clip={avg_clip:.2%}, TOTAL TIME={step_total:.2f}s")

        # Return metrics
        return {
            "loss": avg_loss,
            "tokens": total_tokens,
            "records": num_fresh,
            "clip_fraction": avg_clip,
            "kl": avg_kl,
            "train_step": self.train_step_idx,
            "step_time_s": step_total,
        }, self.train_step_idx

    async def train_step(
        self,
        batch: TrainingBatch,
    ) -> Dict[str, float]:
        """
        Execute one training step with gradient accumulation.

        The CPU/GPU-bound MLX work runs in a separate thread via asyncio.to_thread()
        to avoid blocking the event loop, allowing generation to proceed in parallel.

        Args:
            batch: TrainingBatch containing records to train on

        Returns:
            Dictionary of metrics (loss, tokens, records, stale_dropped, etc.)
        """
        overall_start = time.perf_counter()
        self.model.train()
        verbose = self.config.verbose

        if verbose:
            print(f"\n[train_step {self.train_step_idx + 1}] starting with {len(batch.records)} records")

        # 1. Filter stale records (cheap, do on main thread)
        t0 = time.perf_counter()
        fresh_records = self.filter_stale_records(batch.records)
        stale_count = len(batch.records) - len(fresh_records)
        filter_time = time.perf_counter() - t0

        if verbose and stale_count > 0:
            print(f"  → filtered out {stale_count} stale records (policy lag > {self.config.max_policy_lag})")

        if not fresh_records:
            if verbose:
                print(f"  → skipping step (all records stale)")
            return {
                "skipped": 1,
                "stale_dropped": stale_count,
                "records": 0,
            }

        # 2. Run CPU/GPU-bound training in a separate thread
        # This allows the asyncio event loop to continue processing other tasks
        # (like generation HTTP requests) while training runs
        if verbose:
            print(f"  → dispatching to training thread...")
        t0 = time.perf_counter()
        metrics, new_step_idx = await asyncio.to_thread(
            self._train_step_sync,
            fresh_records,
            verbose,
        )
        training_time = time.perf_counter() - t0
        metrics["stale_dropped"] = stale_count

        # 3. Push weights to inference server (async HTTP, stays on main thread)
        publish_time = 0.0
        if self.publisher is not None:
            if verbose:
                print(f"  → publishing weights (version {new_step_idx}) to {self.publisher.base_url}...")
            t0 = time.perf_counter()
            try:
                await self.publisher.publish(
                    self.model,
                    version=new_step_idx,
                    verbose=verbose,
                )
                publish_time = time.perf_counter() - t0
                if verbose:
                    print(f"  → weights published successfully in {publish_time*1000:.1f}ms")
            except Exception as e:
                publish_time = time.perf_counter() - t0
                if verbose:
                    print(f"  → failed to publish weights after {publish_time*1000:.1f}ms: {e}")

        overall_time = time.perf_counter() - overall_start
        if verbose:
            print(f"  ════════════════════════════════════════════════════════")
            print(f"  TIMING SUMMARY: filter={filter_time*1000:.1f}ms, "
                  f"training={training_time*1000:.1f}ms, "
                  f"publish={publish_time*1000:.1f}ms, "
                  f"TOTAL={overall_time*1000:.1f}ms")
            print(f"  ════════════════════════════════════════════════════════")

        metrics["overall_time_s"] = overall_time
        metrics["publish_time_s"] = publish_time

        return metrics
