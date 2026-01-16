"""
GAN-Style Adversarial Task: Generator vs Discriminator self-play.

This task follows the proposer_solver pattern:
- GeneratorEpisode spawns Monte Carlo DiscriminatorEpisodes
- DiscriminatorEpisode is used both standalone (real data) and as sub-episodes (fake data)
- Both actors are trainable

Structure:
1. GeneratorEpisode.rollout():
   - One model call to generate fake content
   - Spawns N DiscriminatorEpisodes on that fake content (trainable)
   - Generator reward = 1 - pass_rate (where pass_rate = disc correctly identifies fake)
2. DiscriminatorEpisode: Single-turn classification (real vs fake)
3. Arena schedules both:
   - GeneratorEpisodes (fake data, spawn disc sub-episodes)
   - DiscriminatorEpisodes (real data from store)

Output format: Both actors use \\boxed{} for final answer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..core import (
    Messages,
    Rollout,
    Episode,
    EpisodeState,
    SingleTurnEpisode,
    Rubric,
    Arena,
    InferenceClient,
    CreditAssigner,
    Artifact,
    Step,
    EpisodeRequest,
    RAECredit,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Match \boxed{...} with nested braces support
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.IGNORECASE)


def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{} command."""
    match = BOXED_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def parse_judgment(text: str) -> str:
    """
    Parse discriminator judgment from boxed output.

    Returns: "real", "fake", or "invalid"
    """
    boxed = extract_boxed(text)
    if boxed is None:
        return "invalid"

    boxed_lower = boxed.lower()

    # Check for REAL or FAKE in the boxed content
    has_real = "real" in boxed_lower
    has_fake = "fake" in boxed_lower

    if has_real and has_fake:
        return "invalid"
    if has_real:
        return "real"
    if has_fake:
        return "fake"
    return "invalid"


# ---------------------------------------------------------------------------
# Reward Functions
# ---------------------------------------------------------------------------


def discriminator_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Reward for discriminator based on correct classification.

    - Correct classification: +1
    - Wrong classification: 0
    - Invalid format: -0.5
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    judgment = rollout.extras.get("discriminator_judgment", "invalid")
    is_real = rollout.artifact.get("is_real", True)

    if judgment == "invalid":
        if arena.verbose:
            print(f"    [discriminator_reward] invalid format → -0.5")
        return {actor: -0.5}

    # Check if discriminator was correct
    if is_real:
        correct = (judgment == "real")
    else:
        correct = (judgment == "fake")

    reward = 1.0 if correct else 0.0

    if arena.verbose:
        expected = "real" if is_real else "fake"
        status = "CORRECT" if correct else "WRONG"
        print(f"    [discriminator_reward] {status}: judged '{judgment}' for {expected} content → {reward}")

    return {actor: reward}


def generator_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Reward for generator based on discriminator pass rate.

    Generator reward = 1 - pass_rate
    (where pass_rate = fraction of discriminators that correctly identify fake)

    Invalid discriminator judgments are filtered out.
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    generated = rollout.extras.get("generated_content", "")

    # Invalid generation
    if not generated or len(generated.strip()) < 2:
        if arena.verbose:
            print(f"    [generator_reward] invalid/empty generation → -1.0")
        return {actor: -1.0}

    # Get pass rate from extras (computed in get_extras, filtering invalids)
    pass_rate = rollout.extras.get("pass_rate", 0.0)
    n_valid = rollout.extras.get("n_valid_judgments", 0)

    if n_valid == 0:
        # All discriminators gave invalid outputs
        if arena.verbose:
            print(f"    [generator_reward] no valid judgments → 0.0")
        return {actor: 0.0}

    # Generator wants discriminators to be WRONG (say "real" for fake content)
    # So generator reward = 1 - pass_rate
    reward = 1.0 - pass_rate

    if arena.verbose:
        print(f"    [generator_reward] pass_rate={pass_rate:.2f} (n={n_valid}) → reward={reward:.2f}")

    return {actor: reward}


# ---------------------------------------------------------------------------
# Discriminator Episode
# ---------------------------------------------------------------------------


class DiscriminatorEpisode(SingleTurnEpisode):
    """
    Single-turn episode where discriminator classifies content as real or fake.

    Used both:
    - Standalone (real data from store)
    - As sub-episodes spawned by GeneratorEpisode (fake data)

    Output format: Think through reasoning, then put answer in \\boxed{REAL} or \\boxed{FAKE}
    """

    def __init__(self, discriminator_actor_id: str = "Discriminator"):
        super().__init__(max_turns=1)
        self.discriminator_actor_id = discriminator_actor_id
        self._rubric = Rubric(funcs=[discriminator_reward])

    @property
    def episode_type(self) -> str:
        return "discriminate"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        return self.discriminator_actor_id

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        content = artifact.get("content", "")

        return f"""Analyze the following content and determine if it is REAL (authentic human-created) or FAKE (AI-generated).

Content:
{content}

Think step by step about what makes this content appear real or fake. Consider:
- Language patterns and naturalness
- Specificity and detail level
- Consistency and logical coherence
- Any telltale signs of AI generation

After your analysis, provide your final answer in a boxed command.
Your answer must be either \\boxed{{REAL}} or \\boxed{{FAKE}}"""

    def init_state(self, state: EpisodeState, artifact: Any) -> None:
        """Store ground truth in state."""
        state.data["is_real"] = artifact.get("is_real", True)
        state.data["content"] = artifact.get("content", "")

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Return episode data for rubric and preview."""
        completion = state.last_completion_text or ""
        judgment = parse_judgment(completion)
        boxed = extract_boxed(completion)

        return {
            "content": state.data.get("content", ""),
            "is_real": state.data.get("is_real", True),
            "discriminator_judgment": judgment,
            "boxed_output": boxed,
        }


# ---------------------------------------------------------------------------
# Generator Episode
# ---------------------------------------------------------------------------


class GeneratorEpisode(Episode):
    """
    Generator episode that creates fake content and spawns discriminator sub-episodes.

    Flow:
    1. rollout() generates fake content (one model call)
    2. Spawns N discriminator episodes (trainable!)
    3. Discriminator results stored in state.child_results
    4. Generator reward = 1 - pass_rate (filtering invalid judgments)

    Output format: Think through reasoning, then put generated content in \\boxed{}
    """

    def __init__(
        self,
        generator_actor_id: str = "Generator",
        n_discriminator_rollouts: int = 4,
    ):
        self.generator_actor_id = generator_actor_id
        self.n_discriminator_rollouts = n_discriminator_rollouts
        self._rubric = Rubric(funcs=[generator_reward])

    @property
    def episode_type(self) -> str:
        return "generate"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    async def rollout(
        self,
        arena: Arena,
        artifact: Any,
        state: Optional[EpisodeState] = None,
    ) -> EpisodeState:
        if state is None:
            state = EpisodeState()

        # Extract few-shot examples from artifact
        examples = artifact.get("examples", [])

        # Generate fake content
        prompt = self._build_prompt(arena, examples)
        response = await self.call_model(self.generator_actor_id, prompt, arena)

        step = Step(
            actor_id=self.generator_actor_id,
            prompt=prompt,
            completion=response.completion,
            prompt_token_ids=response.prompt_token_ids,
            completion_token_ids=response.completion_token_ids,
            completion_logprobs=response.completion_logprobs,
        )
        state.trajectory.append(step)

        # Parse generated content from boxed output
        generated = extract_boxed(response.text)
        if generated is None:
            # Fallback: use the whole response if no boxed found
            generated = response.text.strip()

        state.data["generated_content"] = generated
        state.data["raw_output"] = response.text

        # Spawn TRAINABLE discriminator episodes on fake content
        if generated and len(generated.strip()) >= 2:
            disc_artifact = {
                "content": generated,
                "is_real": False,  # This is fake content
            }

            requests = [
                EpisodeRequest(
                    episode_type="discriminate",
                    artifact=disc_artifact,
                    is_trainable=True,  # Discriminator sub-episodes ARE trained
                )
                for _ in range(self.n_discriminator_rollouts)
            ]
            results = await arena.generate_rollouts(requests)
            state.child_results.extend(results)

        state.done = True
        return state

    def _build_prompt(self, arena: Arena, examples: List[Dict]) -> Messages:
        """Build prompt for generator with few-shot examples."""
        actor = arena.actors[self.generator_actor_id]

        examples_text = ""
        for i, ex in enumerate(examples, 1):
            content = ex.get("content", "")
            examples_text += f"Example {i}:\n{content}\n\n"

        user_content = f"""Here are some real examples of content:

{examples_text}Your task: Generate a NEW piece of content that is similar in style and quality to the examples above. The content should be realistic and indistinguishable from authentic human-created content.

Think step by step about:
- What makes these examples feel authentic
- What style, tone, and structure to use
- How to create something that feels genuinely human-written

After your reasoning, provide your generated content inside a boxed command.
Format: \\boxed{{your generated content here}}"""

        messages: Messages = []
        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})
        messages.append({"role": "user", "content": user_content})

        return messages

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Include child results summary for rubric."""
        # Get judgment for each child discriminator, filtering invalids
        valid_judgments = []
        all_judgments = []

        for child in state.child_results:
            judgment = child.rollout.extras.get("discriminator_judgment", "invalid")
            all_judgments.append(judgment)

            if judgment != "invalid":
                # For fake content, "correct" = discriminator says "fake"
                correct = (judgment == "fake")
                valid_judgments.append(1.0 if correct else 0.0)

        # Pass rate = fraction of valid judgments that correctly identified fake
        pass_rate = sum(valid_judgments) / len(valid_judgments) if valid_judgments else 0.0

        return {
            "generated_content": state.data.get("generated_content"),
            "raw_output": state.data.get("raw_output"),
            "all_judgments": all_judgments,
            "n_valid_judgments": len(valid_judgments),
            "pass_rate": pass_rate,
        }


# ---------------------------------------------------------------------------
# GAN Arena
# ---------------------------------------------------------------------------


class GANArena(Arena):
    """
    Arena for GAN-style adversarial training.

    Schedules:
    - GeneratorEpisodes: Generate fake content, spawn discriminator sub-episodes
    - DiscriminatorEpisodes: Classify real content from store

    Both episode types produce training data.
    """

    def __init__(
        self,
        client: InferenceClient,
        generator_episodes_per_step: int = 4,
        discriminator_episodes_per_step: int = 4,
        n_discriminator_rollouts: int = 4,
        few_shot_k: int = 3,
        verbose: bool = False,
        credit_assigner: Optional[CreditAssigner] = None,
    ):
        if credit_assigner is None:
            credit_assigner = RAECredit(decay=0.95)
        super().__init__(
            client,
            credit_assigner=credit_assigner,
            verbose=verbose,
        )
        self.generator_episodes_per_step = generator_episodes_per_step
        self.discriminator_episodes_per_step = discriminator_episodes_per_step
        self.n_discriminator_rollouts = n_discriminator_rollouts
        self.few_shot_k = few_shot_k

    def get_batch(self) -> List[EpisodeRequest]:
        """
        Generate batch of episodes.

        Returns:
        - Generator episodes (fake content, spawn disc sub-episodes)
        - Discriminator episodes (real content from store)
        """
        requests: List[EpisodeRequest] = []

        if "real_examples" not in self.stores:
            return []

        store = self.stores["real_examples"]

        if store.count() == 0:
            return []

        # Generator episodes (fake content)
        for _ in range(self.generator_episodes_per_step):
            # Sample few-shot examples for generator
            k = min(self.few_shot_k, store.count())
            samples = store.sample(k=k)
            examples = [s.data for s in samples]

            requests.append(EpisodeRequest(
                episode_type="generate",
                artifact={"examples": examples},
            ))

        # Discriminator episodes (real content)
        for _ in range(self.discriminator_episodes_per_step):
            sample = store.sample_one()
            if sample:
                requests.append(EpisodeRequest(
                    episode_type="discriminate",
                    artifact={
                        "content": sample.data.get("content", ""),
                        "is_real": True,
                    },
                ))

        return requests
