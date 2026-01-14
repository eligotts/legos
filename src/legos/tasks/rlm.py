"""
RLM (Recursive Language Model) Task

Proof of concept demonstrating how the RLM paradigm maps to LEGOS abstractions:
- A single episode type that can recursively spawn itself
- Multi-turn REPL-style interaction
- Hierarchical reward propagation from root to all nested children

This code is illustrative - REPL execution is mocked/commented.
The focus is on the abstraction patterns, not runtime behavior.

RLM Paper: https://arxiv.org/abs/2512.24601
"""

import re
from typing import Any, Dict, List, Tuple

from legos.core.episode import MultiTurnEpisode, EpisodeState
from legos.core.arena import Arena
from legos.core.rubric import Rubric
from legos.core.credit import GRPOCredit
from legos.core.types import Actor, Artifact, EpisodeRequest, Rollout, GenerateResult


# =============================================================================
# Constants
# =============================================================================

RLM_SYSTEM_PROMPT = """You are an RLM (Recursive Language Model) agent.

You have access to a Python REPL with variable `context` containing a long document.
You cannot see context directly - use code to inspect it.

Actions (use exact format):

[CODE]
print(len(context))
print(context[:500])
[/CODE]

[SPAWN]
Sub-task 1
---
Sub-task 2
[/SPAWN]

[ANSWER]
Your final answer
[/ANSWER]
"""


# =============================================================================
# Action Parsing
# =============================================================================

def parse_action(text: str) -> Tuple[str, Any]:
    """
    Parse structured actions from LLM output.

    Returns:
        ("code", code_string) - Execute Python code
        ("spawn", [prompts]) - Launch sub-episodes
        ("answer", answer_string) - Submit final answer
        ("none", None) - No recognized action
    """
    # Priority: answer > spawn > code
    if match := re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', text, re.DOTALL):
        return ("answer", match.group(1).strip())

    if match := re.search(r'\[SPAWN\](.*?)\[/SPAWN\]', text, re.DOTALL):
        prompts = [p.strip() for p in match.group(1).split('---') if p.strip()]
        return ("spawn", prompts)

    if match := re.search(r'\[CODE\](.*?)\[/CODE\]', text, re.DOTALL):
        return ("code", match.group(1).strip())

    return ("none", None)


# =============================================================================
# RLM Episode
# =============================================================================

class RLMEpisode(MultiTurnEpisode):
    """
    RLM Episode - recursively spawns itself for sub-tasks.

    Key patterns demonstrated:
    1. Multi-turn: LLM generates action -> env processes -> repeat
    2. Recursive: [SPAWN] creates new RLMEpisode instances
    3. Trainable children: All spawned episodes enter training batch
    4. Hierarchical reward: Root score propagates to all descendants

    The episode behaves identically whether root or nested - both have
    access to REPL, both can spawn sub-episodes, both can answer.
    """

    episode_type = "rlm"

    def __init__(self, max_turns: int = 10):
        super().__init__(max_turns=max_turns)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def init_state(self, state: EpisodeState, artifact: Any) -> None:
        """
        Initialize episode state.

        artifact schema:
            context: str - The long document (stored in REPL, not shown directly)
            question: str - The task to accomplish
            answer: str - Ground truth for reward computation
            is_root: bool - Whether this is the top-level episode (for rewards)
        """
        state.data["context"] = artifact.get("context", "")
        state.data["question"] = artifact.get("question", "")
        state.data["ground_truth"] = artifact.get("answer", "")
        state.data["is_root"] = artifact.get("is_root", True)
        state.data["final_answer"] = None

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        """Return the actor that starts the episode."""
        return "RLM"

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        """Task description shown at start of episode."""
        ctx_len = len(state.data["context"])
        return f"""Task: {state.data['question']}

Context is {ctx_len:,} characters (in variable `context`).
Use [CODE] to inspect, [SPAWN] to delegate, [ANSWER] when ready."""

    # -------------------------------------------------------------------------
    # Turn Loop
    # -------------------------------------------------------------------------
    async def env_response(
        self,
        state: EpisodeState,
        arena: Arena,
        artifact: Any,
    ) -> str:
        """
        Process the LLM's action and return feedback.

        This is where the magic happens:
        - [CODE] would execute in a REPL (mocked here)
        - [SPAWN] creates new RLMEpisodes via arena.generate_rollouts()
        - [ANSWER] stores the final answer
        """
        text = state.trajectory[-1].completion[0]["content"]
        action, data = parse_action(text)

        if action == "code":
            # NOTE: In production, this would execute in a sandboxed REPL.
            # For this proof of concept, we just acknowledge the code.
            return "[Code executed - output would appear here]"

        elif action == "spawn":
            # Spawn sub-episodes for each prompt in the list
            requests = [
                EpisodeRequest(
                    episode_type="rlm",  # Same type - recursive!
                    artifact={
                        "context": state.data["context"],  # Pass context down, or theoretically do whatever you want here for passing context
                        "question": prompt,
                        "answer": "",  # Sub-tasks don't have ground truth
                        "is_root": False,  # Mark as nested
                    },
                    is_trainable=True,  # Sub-episodes ARE trained
                )
                for prompt in data
            ]
            results = await arena.generate_rollouts(requests)
            state.child_results.extend(results)

            # Collect sub-answers
            answers = []
            for i, r in enumerate(results):
                ans = r.rollout.extras.get("final_answer", "[no answer]")
                answers.append(f"[{i+1}] {ans}")

            return "Sub-results:\n" + "\n".join(answers)

        elif action == "answer":
            state.data["final_answer"] = data
            return "[Answer recorded]"

        return "[No action recognized]"

    def is_done(self, state: EpisodeState, artifact: Any) -> bool:
        """Episode ends when answer submitted or max turns reached."""
        if state.data.get("final_answer") is not None:
            return True
        if self.max_turns > 0 and state.turn >= self.max_turns:
            return True
        return False

    def get_next_actor(self, state: EpisodeState, artifact: Any) -> str:
        """Single actor, always returns RLM."""
        return "RLM"

    # -------------------------------------------------------------------------
    # Rubric Interface
    # -------------------------------------------------------------------------

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        """Data passed to rubric for reward computation."""
        return {
            "final_answer": state.data.get("final_answer", ""),
            "ground_truth": state.data.get("ground_truth", ""),
            "is_root": state.data.get("is_root", True),
            "child_results": state.child_results,  # For reward propagation
        }

    @property
    def rubric(self) -> Rubric:
        return Rubric([hierarchical_reward])


# =============================================================================
# Reward Function
# =============================================================================

def hierarchical_reward(rollout: Rollout, arena: Arena) -> Dict[str, float]:
    """
    Hierarchical reward: root computes correctness, propagates to all children.

    This demonstrates how a tree of episodes can share a single reward signal:
    1. Non-root episodes return 0 (placeholder)
    2. Root computes correctness
    3. Root traverses child_results tree, setting rewards on all descendants

    The result: every step in every nested episode gets the same reward
    as the root, creating aligned training signal across the recursion.
    """
    is_root = rollout.extras.get("is_root", True)

    if not is_root:
        # Non-root: return 0, will be overwritten by ancestor
        return {"RLM": 0.0}

    # Root: compute correctness
    final = rollout.extras.get("final_answer", "").strip().lower()
    truth = rollout.extras.get("ground_truth", "").strip().lower()
    reward = 1.0 if (final == truth or truth in final) else 0.0

    # Propagate reward to all descendants
    def propagate(children: List[GenerateResult], value: float) -> None:
        for child in children:
            child.rollout.rewards["RLM"] = value
            for step in child.rollout.steps:
                step.reward = value
            propagate(child.children, value)

    propagate(rollout.extras.get("child_results", []), reward)

    return {"RLM": reward}


# =============================================================================
# Arena
# =============================================================================

class RLMArena(Arena):
    """
    Arena for RLM training.

    Demonstrates:
    - Single actor registration
    - Single recursive episode type
    - GRPO-style batching (same task N times)
    """

    def __init__(
        self,
        client,
        data: List[Dict],
        episodes_per_step: int = 4,
    ):
        super().__init__(client, credit_assigner=GRPOCredit())
        self.episodes_per_step = episodes_per_step

        # One actor
        self.add_actor(Actor(id="RLM", system_prompt=RLM_SYSTEM_PROMPT))

        # One episode type (recursive)
        self.add_episode("rlm", RLMEpisode())

        # Data store
        self.add_store("tasks")
        for item in data:
            self.stores["tasks"].add(Artifact(data=item))

    def get_batch(self) -> List[EpisodeRequest]:
        """Sample one task, run N times for GRPO advantage estimation."""
        sample = self.stores["tasks"].sample_one()
        return [
            EpisodeRequest(
                episode_type="rlm",
                artifact={
                    "context": sample.data["context"],
                    "question": sample.data["question"],
                    "answer": sample.data["answer"],
                    "is_root": True,
                },
            )
            for _ in range(self.episodes_per_step)
        ]
