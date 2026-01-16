"""
Text Reversal: Proposer/Solver task where both actors are trainable.

This task trains:
- Proposer: Generates strings to be reversed (learns to produce diverse strings)
- Solver: Reverses strings (learns the reversal task)

Key difference from proposer_solver.py:
- Solver episodes are `is_trainable=True` so both actors learn
- Arena only schedules proposer episodes (solvers spawn inside)

Structure:
1. TextReversalProposerEpisode.rollout():
   - One model call to generate a string
   - Spawns N trainable solver episodes
   - Collects results for scoring
2. ReverseEpisode: Single-turn solve attempt (trainable)
3. Rewards: Solver gets exact match, Proposer gets goldilocks reward
"""
from __future__ import annotations

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
    Step,
    EpisodeRequest,
    GRPOCredit,
    CreditAssigner,
)


# ---------------------------------------------------------------------------
# Reward Functions
# ---------------------------------------------------------------------------


def reverse_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Exact match reward for solver.

    Returns 1.0 if the predicted reversal matches ground truth, 0.0 otherwise.
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    predicted = rollout.steps[-1].completion_text.strip() if rollout.steps else ""
    ground_truth = rollout.artifact.get("ground_truth", "")

    correct = predicted == ground_truth
    reward = 1.0 if correct else 0.0

    if arena.verbose:
        status = "CORRECT" if correct else "WRONG"
        print(f"    [reverse_reward] {status}: '{predicted[:30]}' vs '{ground_truth[:30]}' → {reward}")

    return {actor: reward}


def proposer_reward(
    rollout: Rollout,
    arena: Arena,
    target_pass_rate: float = 0.5,
) -> Dict[str, float]:
    """
    Goldilocks reward for proposer based on solver pass rate.

    Target ~50% pass rate (not too easy, not too hard).
    This encourages the proposer to generate interesting/diverse strings.
    """
    if not rollout.actors:
        return {}

    actor = next(iter(rollout.actors))
    proposed = rollout.extras.get("proposed_string", "")

    # Invalid or too short string
    if not proposed or len(proposed) < 2:
        if arena.verbose:
            print(f"    [proposer_reward] invalid/short string → -1.0")
        return {actor: -1.0}

    # Get pass rate from extras (computed in get_extras)
    pass_rate = rollout.extras.get("pass_rate", 0.0)

    # Reward peaks at target pass rate
    distance = abs(pass_rate - target_pass_rate)
    reward = 1.0 - (distance * 2)  # Max 1.0 at target, 0.0 at extremes
    final_reward = max(-0.5, reward)

    if arena.verbose:
        print(f"    [proposer_reward] pass_rate={pass_rate:.2f}, target={target_pass_rate}, reward={final_reward:.2f}")

    return {actor: final_reward}


# ---------------------------------------------------------------------------
# Reverse Episode (Solver)
# ---------------------------------------------------------------------------


class ReverseEpisode(SingleTurnEpisode):
    """
    Single-turn episode where the Solver reverses a string.

    Spawned by TextReversalProposerEpisode as a trainable sub-episode.
    """

    def __init__(self, solver_actor_id: str = "Solver"):
        super().__init__(max_turns=1)
        self.solver_actor_id = solver_actor_id
        self._rubric = Rubric(funcs=[reverse_reward])

    @property
    def episode_type(self) -> str:
        return "reverse"

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        return self.solver_actor_id

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        string = artifact.get("string", "")

        return f"""Reverse the following string character by character.

String: {string}

Output ONLY the reversed string, nothing else. Do not add any explanation or extra text."""

    def init_state(self, state: EpisodeState, artifact: Any) -> None:
        """Store the input string and ground truth in state."""
        state.data["string"] = artifact.get("string", "")
        state.data["ground_truth"] = artifact.get("ground_truth", "")

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Include response and extracted answer for preview/debugging."""
        response = state.last_completion_text or ""
        predicted = response.strip()

        return {
            "string": state.data.get("string", ""),
            "ground_truth": state.data.get("ground_truth", ""),
            "predicted": predicted,
            "correct": predicted == state.data.get("ground_truth", ""),
        }


# ---------------------------------------------------------------------------
# Text Reversal Proposer Episode
# ---------------------------------------------------------------------------


class TextReversalProposerEpisode(Episode):
    """
    Proposer episode that generates strings and spawns trainable solver episodes.

    Flow:
    1. rollout() generates a string (one model call)
    2. Computes ground truth by reversing the string
    3. Spawns N solver episodes (trainable!)
    4. Solver results stored in state.child_results
    5. Rubric uses extras to compute proposer reward based on pass rate
    """

    def __init__(
        self,
        proposer_actor_id: str = "Proposer",
        n_solver_rollouts: int = 4,
        target_pass_rate: float = 0.5,
    ):
        self.proposer_actor_id = proposer_actor_id
        self.n_solver_rollouts = n_solver_rollouts
        self.target_pass_rate = target_pass_rate
        self._rubric = Rubric(funcs=[proposer_reward])

    @property
    def episode_type(self) -> str:
        return "text_reversal"

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

        # Generate string
        prompt = self._build_prompt(arena)
        response = await self.call_model(self.proposer_actor_id, prompt, arena)

        step = Step(
            actor_id=self.proposer_actor_id,
            prompt=prompt,
            completion=response.completion,
            prompt_token_ids=response.prompt_token_ids,
            completion_token_ids=response.completion_token_ids,
            completion_logprobs=response.completion_logprobs,
        )
        state.trajectory.append(step)

        # Parse proposed string and compute ground truth
        proposed_string = response.text.strip()
        ground_truth = proposed_string[::-1]  # Python string reversal
        state.data["proposed_string"] = proposed_string
        state.data["ground_truth"] = ground_truth

        # Spawn TRAINABLE solver episodes
        if proposed_string and len(proposed_string) >= 2:
            solver_artifact = {
                "string": proposed_string,
                "ground_truth": ground_truth,
            }

            requests = [
                EpisodeRequest(
                    episode_type="reverse",
                    artifact=solver_artifact,
                    is_trainable=True,  # KEY: Solvers ARE trained
                )
                for _ in range(self.n_solver_rollouts)
            ]
            results = await arena.generate_rollouts(requests)
            state.child_results.extend(results)

        state.done = True
        return state

    def _build_prompt(self, arena: Arena) -> Messages:
        """Build the prompt for the proposer."""
        actor = arena.actors[self.proposer_actor_id]

        user_content = """Generate a string of text to be reversed. The string should be:
- Between 5 and 50 characters long
- Can contain letters, numbers, spaces, and basic punctuation
- Should be interesting and varied (not just "hello" or "test")

Output ONLY the string, nothing else. Do not add any explanation or extra text."""

        messages: Messages = []
        if actor.system_prompt:
            messages.append({"role": "system", "content": actor.system_prompt})
        messages.append({"role": "user", "content": user_content})

        return messages

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Include child results summary for rubric."""
        # Get reward for each child solver
        child_rewards = []
        for child in state.child_results:
            actor = next(iter(child.rollout.actors)) if child.rollout.actors else None
            reward = child.rewards.get(actor, 0.0) if actor else 0.0
            child_rewards.append(reward)

        pass_rate = sum(1 for r in child_rewards if r > 0.5) / len(child_rewards) if child_rewards else 0.0

        return {
            "proposed_string": state.data.get("proposed_string"),
            "ground_truth": state.data.get("ground_truth"),
            "proposed_raw": state.last_completion_text,
            "solver_rewards": child_rewards,
            "pass_rate": pass_rate,
            "n_solvers": len(child_rewards),
        }


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------


class TextReversalArena(Arena):
    """
    Arena that schedules proposer episodes only.

    Solver episodes are spawned inside proposer episodes as trainable sub-episodes.
    Both proposer and solver actors are trained.
    """

    def __init__(
        self,
        client: InferenceClient,
        episodes_per_step: int = 4,
        verbose: bool = False,
        credit_assigner: Optional[CreditAssigner] = None,
    ):
        if credit_assigner is None:
            credit_assigner = GRPOCredit()
        super().__init__(client, credit_assigner=credit_assigner, verbose=verbose)
        self.episodes_per_step = episodes_per_step

    def get_batch(self) -> List[EpisodeRequest]:
        """
        Only schedule proposer episodes.

        Solver episodes are spawned inside proposer episodes as trainable children.
        """
        return [
            EpisodeRequest(episode_type="text_reversal", artifact={})
            for _ in range(self.episodes_per_step)
        ]
