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

from legos.core.episode import MultiTurnEpisode, SingleTurnEpisode, EpisodeState
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

SUMMARIZER_SYSTEM_PROMPT = """You are a context summarizer.
Condense the conversation history into a concise summary preserving:
- Key facts and findings
- Current progress on the task
- Important context for continuing
Be brief but complete."""


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
    5. Context continuation: When context threshold hit, spawns summarizer
       and continues in a new episode segment linked via .next

    The episode behaves identically whether root or nested - both have
    access to REPL, both can spawn sub-episodes, both can answer.
    """

    episode_type = "rlm"

    def __init__(self, max_turns: int = 10, context_threshold: int = 4000):
        super().__init__(max_turns=max_turns)
        self.context_threshold = context_threshold

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
        - Context threshold detection triggers summarization and continuation
        """
        text = state.trajectory[-1].completion[0]["content"]
        action, data = parse_action(text)

        response_str = ""

        if action == "code":
            # NOTE: In production, this would execute in a sandboxed REPL.
            # For this proof of concept, we just acknowledge the code.
            response_str = "[Code executed - output would appear here]"

        elif action == "spawn":
            # Spawn sub-episodes for each prompt in the list
            requests = [
                EpisodeRequest(
                    episode_type="rlm",  # Same type - recursive!
                    artifact={
                        "context": state.data["context"],  # Pass context down
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

            response_str = "Sub-results:\n" + "\n".join(answers)

        elif action == "answer":
            state.data["final_answer"] = data
            response_str = "[Answer recorded]"

        else:
            response_str = "[No action recognized]"

        # Check context threshold after each turn
        context_tokens = self._estimate_context_tokens(state)
        if context_tokens > self.context_threshold:
            # Spawn summarizer as child (trainable, will get same reward via hierarchical_reward)
            summary_request = EpisodeRequest(
                episode_type="summarize",
                artifact={
                    "question": state.data["question"],
                    "history": self._build_history(state),
                },
                is_trainable=True,
            )
            summary_results = await arena.generate_rollouts([summary_request])
            state.child_results.extend(summary_results)

            summary_text = summary_results[0].rollout.extras.get("summary", "")

            # Request continuation with summarized context
            # Pass through is_root status - if we're root, continuation maintains that
            state.continuation_request = {
                "episode_type": "rlm",
                "artifact": {
                    "context": summary_text,  # Summarized context replaces full history
                    "question": state.data["question"],
                    "answer": state.data["ground_truth"],
                    "is_root": state.data.get("is_root", True),
                },
            }
            state.done = True
            return "[Context limit reached, summarizing and continuing...]"

        return response_str

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

    # -------------------------------------------------------------------------
    # Context Continuation (Override generate to handle .next chaining)
    # -------------------------------------------------------------------------

    async def generate(
        self,
        arena: Arena,
        artifact: Any,
        meta: Dict[str, Any] = None,
        is_trainable: bool = True,
    ) -> GenerateResult:
        """
        Override to handle continuation before scoring.

        When context threshold is hit during rollout:
        1. Spawns Summarizer as child (trainable, will get same reward)
        2. Sets continuation_request for new segment
        3. After rollout, launches continuation and links via .next
        4. Each segment scores independently with same final_answer -> same reward
        """
        import time

        # Create state and call init_state for initialization
        state = EpisodeState()
        self.init_state(state, artifact)

        state = await self.rollout(arena, artifact, state)

        # Handle continuation BEFORE building rollout extras
        continuation = None
        final_answer = state.data.get("final_answer")

        if state.continuation_request is not None:
            # Launch continuation - it handles its own scoring recursively
            continuation = await arena.run_episode(
                state.continuation_request["episode_type"],
                state.continuation_request["artifact"],
                meta=meta,
                is_trainable=is_trainable,
            )
            # Grab final_answer from continuation's rollout (it already has it set)
            final_answer = continuation.rollout.extras.get("final_answer")

        # Build rollout with final_answer (whether from us or continuation)
        is_root = state.data.get("is_root", True)
        rollout = Rollout(
            episode_type=self.episode_type,
            artifact=artifact,
            meta=meta or {},
            steps=state.trajectory,
            extras={
                "final_answer": final_answer,
                "ground_truth": state.data.get("ground_truth", ""),
                "is_root": is_root,
                "child_results": state.child_results,
            },
            ended_at=time.time(),
        )

        # Score - rubric computes reward and propagates to children
        await self.rubric.score(rollout, arena)

        return GenerateResult(
            rollout=rollout,
            children=state.child_results,
            next=continuation,
            is_trainable=is_trainable,
        )

    def _estimate_context_tokens(self, state: EpisodeState) -> int:
        """Estimate current context size in tokens (rough: 4 chars per token)."""
        total_chars = sum(
            len(step.prompt_text) + len(step.completion_text)
            for step in state.trajectory
        )
        return total_chars // 4

    def _build_history(self, state: EpisodeState) -> str:
        """Build conversation history string for summarization."""
        lines = []
        for i, step in enumerate(state.trajectory):
            lines.append(f"Turn {i+1}: {step.completion_text[:500]}...")
        return "\n".join(lines)


# =============================================================================
# Summarization Episode
# =============================================================================

class SummarizationEpisode(SingleTurnEpisode):
    """
    Episode for summarizing conversation history.

    Used when RLMEpisode hits context threshold to compress history
    before continuing in a new segment.
    """

    episode_type = "summarize"

    def get_initial_actor(self, artifact: Any, state: EpisodeState) -> str:
        return "Summarizer"

    def get_initial_prompt(
        self,
        arena: Arena,
        artifact: Any,
        state: EpisodeState,
    ) -> str:
        question = artifact.get("question", "")
        history = artifact.get("history", "")
        return f"""Summarize this conversation for continuation:

Task: {question}

History:
{history}

Provide a concise summary that preserves key facts, findings, and progress."""

    def get_extras(self, state: EpisodeState) -> Dict[str, Any]:
        return {"summary": state.last_completion_text}

    @property
    def rubric(self) -> Rubric:
        # Summarizer gets reward via hierarchical propagation from parent RLM
        return Rubric([lambda rollout, arena: {"Summarizer": 0.0}])


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

    Note: This propagates to both RLM sub-episodes and Summarizer episodes.
    """
    is_root = rollout.extras.get("is_root", True)

    if not is_root:
        # Non-root: return 0, will be overwritten by ancestor
        return {"RLM": 0.0}

    # Root: compute correctness
    final = rollout.extras.get("final_answer", "").strip().lower()
    truth = rollout.extras.get("ground_truth", "").strip().lower()
    reward = 1.0 if (final == truth or truth in final) else 0.0

    # Propagate reward to all descendants (including Summarizer children)
    def propagate(children: List[GenerateResult], value: float) -> None:
        for child in children:
            # Get the actor IDs from steps in this child rollout
            for step in child.rollout.steps:
                child.rollout.rewards[step.actor_id] = value
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
    - Actor registration (RLM + Summarizer)
    - Recursive episode type with context continuation
    - GRPO-style batching (same task N times)
    """

    def __init__(
        self,
        client,
        data: List[Dict],
        episodes_per_step: int = 4,
        context_threshold: int = 4000,
    ):
        super().__init__(client, credit_assigner=GRPOCredit())
        self.episodes_per_step = episodes_per_step

        # Actors
        self.add_actor(Actor(id="RLM", system_prompt=RLM_SYSTEM_PROMPT))
        self.add_actor(Actor(id="Summarizer", system_prompt=SUMMARIZER_SYSTEM_PROMPT))

        # Episode types
        self.add_episode("rlm", RLMEpisode(context_threshold=context_threshold))
        self.add_episode("summarize", SummarizationEpisode())

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
