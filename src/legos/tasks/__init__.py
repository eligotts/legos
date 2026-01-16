"""
Task implementations for legos.

These tasks demonstrate how to implement various self-play scenarios:
1. Proposer/Solver: AZR-style problem generation and solving
2. GSM8K: Math reasoning with GRPO-style training
3. Negotiation: SPIRAL-style SimpleNegotiation with RAE credit assignment
4. RefineLoop: Generator-Critic iterative refinement (cooperative)
5. HeadToHead: Tournament competition with LLM-judged matches
6. SPICE: Self-Play In Corpus Environment (corpus-grounded Q&A generation)
7. RLM: Recursive Language Model (recursive self-spawning episodes)
8. GAN: Generator vs Discriminator adversarial training
9. TextReversal: Proposer/Solver with trainable solvers (string reversal)

See examples/train_*.py scripts for complete usage.
"""
from .proposer_solver import (
    ProposerEpisode,
    SolveEpisode,
    ProposerSolverArena,
)

from .gsm8k import (
    GSM8KEpisode,
    GSM8KArena,
    gsm8k_reward,
    load_gsm8k,
    extract_xml_answer,
    extract_hash_answer,
)

from .negotiation import (
    NegotiationEpisode,
    NegotiationArena,
    negotiation_reward,
)

from .refine_loop import (
    RefineLoopEpisode,
    RefineLoopArena,
    refine_loop_reward,
)

from .head_to_head import (
    MatchEpisode,
    ChallengeProposerEpisode,
    HeadToHeadArena,
    match_reward,
)

from .spice import (
    SpiceProposerEpisode,
    SpiceSolverEpisode,
    SpiceArena,
    solver_llm_judge_reward,
    proposer_pass_rate_reward,
)

from .rlm import (
    RLMEpisode,
    RLMArena,
    hierarchical_reward,
)

from .gan import (
    GeneratorEpisode,
    DiscriminatorEpisode,
    GANArena,
    generator_reward,
    discriminator_reward,
    parse_judgment,
    extract_boxed,
)

from .text_reversal import (
    ReverseEpisode,
    TextReversalProposerEpisode,
    TextReversalArena,
    reverse_reward,
    proposer_reward,
)

__all__ = [
    # Proposer/Solver
    "ProposerEpisode",
    "SolveEpisode",
    "ProposerSolverArena",
    # GSM8K
    "GSM8KEpisode",
    "GSM8KArena",
    "gsm8k_reward",
    "load_gsm8k",
    "extract_xml_answer",
    "extract_hash_answer",
    "GSM8K_SYSTEM_PROMPT",
    # Negotiation (SPIRAL SimpleNegotiation)
    "NegotiationEpisode",
    "NegotiationArena",
    "negotiation_reward",
    # RefineLoop (generator-critic refinement)
    "RefineLoopEpisode",
    "RefineLoopArena",
    "refine_loop_reward",
    # HeadToHead (tournament competition)
    "MatchEpisode",
    "ChallengeProposerEpisode",
    "HeadToHeadArena",
    "match_reward",
    # SPICE (corpus-grounded Q&A)
    "SpiceProposerEpisode",
    "SpiceSolverEpisode",
    "SpiceArena",
    "solver_llm_judge_reward",
    "proposer_pass_rate_reward",
    # RLM (Recursive Language Model)
    "RLMEpisode",
    "RLMArena",
    "hierarchical_reward",
    # GAN (Generator vs Discriminator)
    "GeneratorEpisode",
    "DiscriminatorEpisode",
    "GANArena",
    "generator_reward",
    "discriminator_reward",
    "parse_judgment",
    "extract_boxed",
    # TextReversal (Proposer/Solver with trainable solvers)
    "ReverseEpisode",
    "TextReversalProposerEpisode",
    "TextReversalArena",
    "reverse_reward",
    "proposer_reward",
]
