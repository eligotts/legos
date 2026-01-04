"""
Task implementations for the self-play engine.

These tasks demonstrate how to implement various self-play scenarios:
1. Debate: Multi-turn, zero-sum competitive dialogue
2. Proposer/Solver: AZR-style problem generation and solving
3. Dataset RL: Standard RL with fixed datasets

See examples/train_*.py scripts for complete usage.
"""
from .debate import (
    DebateEpisode,
    DebateArena,
)

from .proposer_solver import (
    ProposerEpisode,
    SolveEpisode,
    ProposerSolverArena,
)

from .dataset_rl import (
    DatasetEpisode,
    DatasetArena,
    exact_match,
    llm_judge,
    make_llm_judge,
    load_dataset,
)

__all__ = [
    # Debate
    "DebateEpisode",
    "DebateArena",
    # Proposer/Solver
    "ProposerEpisode",
    "SolveEpisode",
    "ProposerSolverArena",
    # Dataset RL
    "DatasetEpisode",
    "DatasetArena",
    "exact_match",
    "llm_judge",
    "make_llm_judge",
    "load_dataset",
]
