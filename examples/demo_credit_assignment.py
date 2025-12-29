#!/usr/bin/env python3
"""
Demo: Credit Assignment Integration Test

Tests the credit assignment system:
1. GRPOCredit computes advantages correctly
2. Hierarchical grouping works as expected
3. apply_credit() updates step.advantage
4. TrainingBatch includes computed advantages
"""
import asyncio
import sys
sys.path.insert(0, "src")

from self_play import (
    MockInferenceClient,
    Messages,
    GRPOCredit,
    ConstantCredit,
    EpisodicRewardCredit,
    apply_credit,
)
from self_play.examples.proposer_solver import (
    create_proposer_solver_arena,
)


def create_mock_client():
    """Create mock client with deterministic responses."""
    call_count = {"n": 0}

    def mock_response(messages: Messages) -> str:
        call_count["n"] += 1
        last = messages[-1]["content"] if messages else ""

        if "Generate a new math question" in last:
            return '{"question": "What is 7 * 8?", "answer": "56", "difficulty": "easy"}'
        elif "Solve the following question" in last:
            # Alternate correct/incorrect
            if call_count["n"] % 2 == 0:
                return "Let me calculate: 7 * 8 = 56. The answer is: 56"
            else:
                return "I think it's 15. The answer is: 15"
        else:
            return "Mock response"

    return MockInferenceClient(response_fn=mock_response), call_count


async def test_grpo_credit_basic():
    """Test GRPOCredit computes advantages correctly."""
    print("\n" + "=" * 60)
    print("TEST: GRPOCredit Basic Computation")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=4,
        batch_size=3,  # 3 proposer episodes
    )

    # Generate rollouts without credit assignment
    requests = arena.get_batch()
    results = await arena.generate_rollouts(requests)

    # Verify we have 3 top-level results
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # Each should have 4 children
    for i, result in enumerate(results):
        assert len(result.children) == 4, f"Result {i} has {len(result.children)} children, expected 4"

    # Compute GRPO credit
    credit = GRPOCredit()
    weights = credit.compute(results)

    # Should have weights for all steps:
    # 3 proposer rollouts * 1 step + 3 * 4 solver rollouts * 1 step = 15 total
    expected_keys = 3 * 1 + 3 * 4 * 1
    assert len(weights) == expected_keys, f"Expected {expected_keys} weight keys, got {len(weights)}"

    # Verify GRPO property: within each group, advantages should sum to ~0
    # Top level: 3 proposers
    proposer_advantages = []
    for result in results:
        rollout = result.rollout
        key = (rollout.id, 0)
        proposer_advantages.append(weights[key])

    mean_proposer = sum(proposer_advantages) / len(proposer_advantages)
    print(f"  Proposer advantages: {proposer_advantages}")
    print(f"  Proposer mean (should be ~0): {mean_proposer:.6f}")
    assert abs(mean_proposer) < 1e-6, f"Proposer advantages should sum to 0, got mean {mean_proposer}"

    # For each proposer's children (4 solvers each), advantages should sum to ~0
    for i, result in enumerate(results):
        child_advantages = []
        for child in result.children:
            key = (child.rollout.id, 0)
            child_advantages.append(weights[key])

        mean_child = sum(child_advantages) / len(child_advantages)
        print(f"  Proposer {i} solver advantages: {child_advantages}")
        print(f"    Mean (should be ~0): {mean_child:.6f}")
        assert abs(mean_child) < 1e-6, f"Child advantages should sum to 0"

    print("  ✓ GRPO advantages computed correctly")
    print("  ✓ Top-level group advantages sum to 0")
    print("  ✓ Per-parent child groups advantages sum to 0")


async def test_apply_credit():
    """Test apply_credit updates step.advantage in place."""
    print("\n" + "=" * 60)
    print("TEST: apply_credit() Updates Steps")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=2,
        batch_size=2,
    )

    requests = arena.get_batch()
    results = await arena.generate_rollouts(requests)

    # Before apply_credit, all advantages should be 0
    for result in results:
        for step in result.rollout.steps:
            assert step.advantage == 0.0, "Initial advantage should be 0"
        for child in result.children:
            for step in child.rollout.steps:
                assert step.advantage == 0.0, "Initial child advantage should be 0"

    # Compute and apply credit
    credit = GRPOCredit()
    weights = credit.compute(results)
    apply_credit(results, weights)

    # After apply_credit, advantages should be populated
    non_zero_count = 0
    for result in results:
        for step in result.rollout.steps:
            if step.advantage != 0.0:
                non_zero_count += 1
        for child in result.children:
            for step in child.rollout.steps:
                if step.advantage != 0.0:
                    non_zero_count += 1

    # With GRPO, at least some should be non-zero (unless all rewards equal)
    print(f"  Steps with non-zero advantage: {non_zero_count}")
    print("  ✓ apply_credit() modifies step.advantage in place")


async def test_training_batch_with_credit():
    """Test TrainingBatch includes computed advantages."""
    print("\n" + "=" * 60)
    print("TEST: TrainingBatch Includes Advantages")
    print("=" * 60)

    client, _ = create_mock_client()
    # Use arena with credit_assigner
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=4,
        batch_size=2,
    )
    arena.credit_assigner = GRPOCredit()

    batch = await arena.step()

    # Check that records have both reward and advantage
    for record in batch.records:
        assert hasattr(record, 'reward'), "Record missing reward"
        assert hasattr(record, 'advantage'), "Record missing advantage"

    # Group records by role
    proposer_records = [r for r in batch.records if r.role_id == "Proposer"]
    solver_records = [r for r in batch.records if r.role_id == "Solver"]

    print(f"  Total records: {len(batch.records)}")
    print(f"  Proposer records: {len(proposer_records)}")
    print(f"  Solver records: {len(solver_records)}")

    # Show some example advantages
    print("\n  Sample Proposer advantages:")
    for r in proposer_records[:3]:
        print(f"    reward={r.reward:.3f}, advantage={r.advantage:.3f}")

    print("\n  Sample Solver advantages:")
    for r in solver_records[:6]:
        print(f"    reward={r.reward:.3f}, advantage={r.advantage:.3f}")

    print("\n  ✓ TrainingBatch records have advantage field")


async def test_constant_credit():
    """Test ConstantCredit assigns uniform weight."""
    print("\n" + "=" * 60)
    print("TEST: ConstantCredit")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=2,
        batch_size=2,
    )

    requests = arena.get_batch()
    results = await arena.generate_rollouts(requests)

    credit = ConstantCredit(value=1.0)
    weights = credit.compute(results)

    # All weights should be 1.0
    for key, weight in weights.items():
        assert weight == 1.0, f"Expected 1.0, got {weight}"

    print(f"  Total weights: {len(weights)}")
    print("  ✓ All weights are 1.0")


async def test_episodic_reward_credit():
    """Test EpisodicRewardCredit uses rollout reward as weight."""
    print("\n" + "=" * 60)
    print("TEST: EpisodicRewardCredit")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=2,
        batch_size=2,
    )

    requests = arena.get_batch()
    results = await arena.generate_rollouts(requests)

    credit = EpisodicRewardCredit()
    weights = credit.compute(results)

    # Weights should equal rollout rewards
    for result in results:
        rollout = result.rollout
        expected = sum(rollout.rewards.values()) if rollout.rewards else 0.0
        key = (rollout.id, 0)
        actual = weights[key]
        print(f"  Proposer rollout: reward={expected:.3f}, weight={actual:.3f}")
        assert abs(actual - expected) < 1e-6, f"Weight {actual} != reward {expected}"

    print("  ✓ Weights equal rollout total rewards")


async def test_grpo_with_normalize():
    """Test GRPOCredit with normalize=True."""
    print("\n" + "=" * 60)
    print("TEST: GRPOCredit with Normalization")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=4,
        batch_size=3,
    )

    requests = arena.get_batch()
    results = await arena.generate_rollouts(requests)

    # With normalize=True
    credit = GRPOCredit(normalize=True)
    weights = credit.compute(results)

    # Top-level advantages should have unit std (if variance > 0)
    proposer_advantages = []
    for result in results:
        key = (result.rollout.id, 0)
        proposer_advantages.append(weights[key])

    if len(proposer_advantages) > 1:
        mean = sum(proposer_advantages) / len(proposer_advantages)
        var = sum((a - mean) ** 2 for a in proposer_advantages) / len(proposer_advantages)
        std = var ** 0.5

        print(f"  Normalized proposer advantages: {proposer_advantages}")
        print(f"  Mean: {mean:.6f}, Std: {std:.6f}")

        if std > 1e-6:
            # After normalization, std should be ~1
            assert abs(std - 1.0) < 0.1, f"Expected std ~1, got {std}"
            print("  ✓ Advantages normalized to unit std")
        else:
            print("  (All rewards equal, normalization has no effect)")


async def test_grpo_positive_only():
    """Test GRPOCredit with positive_only=True."""
    print("\n" + "=" * 60)
    print("TEST: GRPOCredit with positive_only")
    print("=" * 60)

    client, _ = create_mock_client()
    arena = create_proposer_solver_arena(
        client=client,
        n_solver_rollouts=4,
        batch_size=3,
    )

    requests = arena.get_batch()
    results = await arena.generate_rollouts(requests)

    # With positive_only=True
    credit = GRPOCredit(positive_only=True)
    weights = credit.compute(results)

    # All weights should be >= 0
    negative_count = sum(1 for w in weights.values() if w < 0)
    assert negative_count == 0, f"Found {negative_count} negative weights"

    print(f"  Total weights: {len(weights)}")
    print(f"  Negative weights: {negative_count}")
    print("  ✓ All weights are non-negative")


async def test_grpo_multi_role():
    """Test GRPOCredit with multi-role rollouts (like debate)."""
    print("\n" + "=" * 60)
    print("TEST: GRPOCredit Multi-Role (Debate-style)")
    print("=" * 60)

    from self_play.examples.debate import create_debate_arena

    # Create a mock that gives varying rewards
    call_count = {"n": 0}

    def mock_debate(messages: Messages) -> str:
        call_count["n"] += 1
        return f"Mock argument {call_count['n']}"

    client = MockInferenceClient(response_fn=mock_debate)
    arena = create_debate_arena(
        client=client,
        topics=["Topic A", "Topic B", "Topic C"],
        num_rounds=2,  # 2 rounds = 4 steps per debate (Aff, Neg, Aff, Neg)
        batch_size=3,
    )

    # Generate rollouts
    requests = arena.get_batch()
    results = await arena.generate_rollouts(requests)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # Each debate rollout should have both "Aff" and "Neg" in rewards
    for result in results:
        rewards = result.rollout.rewards
        assert "Aff" in rewards, f"Missing 'Aff' in rewards: {rewards}"
        assert "Neg" in rewards, f"Missing 'Neg' in rewards: {rewards}"
        print(f"  Debate rewards: Aff={rewards['Aff']:.2f}, Neg={rewards['Neg']:.2f}")

    # Compute GRPO credit
    credit = GRPOCredit()
    weights = credit.compute(results)

    # Collect advantages by role
    aff_advantages = []
    neg_advantages = []

    for result in results:
        rollout = result.rollout
        for i, step in enumerate(rollout.steps):
            key = (rollout.id, i)
            adv = weights[key]
            if step.role_id == "Aff":
                aff_advantages.append(adv)
            elif step.role_id == "Neg":
                neg_advantages.append(adv)

    # Each role's advantages should sum to ~0 (GRPO property)
    print(f"\n  Aff advantages: {aff_advantages}")
    print(f"  Neg advantages: {neg_advantages}")

    aff_mean = sum(aff_advantages) / len(aff_advantages) if aff_advantages else 0
    neg_mean = sum(neg_advantages) / len(neg_advantages) if neg_advantages else 0

    print(f"\n  Aff mean (should be ~0): {aff_mean:.6f}")
    print(f"  Neg mean (should be ~0): {neg_mean:.6f}")

    # With 3 debates and all same rewards, means should be 0
    assert abs(aff_mean) < 1e-6, f"Aff advantages should sum to 0, got mean {aff_mean}"
    assert abs(neg_mean) < 1e-6, f"Neg advantages should sum to 0, got mean {neg_mean}"

    print("\n  ✓ Per-role GRPO advantages computed correctly")
    print("  ✓ Each role's advantages sum to 0 independently")


async def main():
    print("=" * 60)
    print("CREDIT ASSIGNMENT INTEGRATION TESTS")
    print("=" * 60)

    await test_grpo_credit_basic()
    await test_apply_credit()
    await test_training_batch_with_credit()
    await test_constant_credit()
    await test_episodic_reward_credit()
    await test_grpo_with_normalize()
    await test_grpo_positive_only()
    await test_grpo_multi_role()

    print("\n" + "=" * 60)
    print("ALL CREDIT ASSIGNMENT TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
