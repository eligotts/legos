"""
Fused LM Head with Chunked Logprob Computation for MLX.

This module provides memory-efficient computation of log-probabilities and entropy
without materializing full [batch, seq_len, vocab_size] tensors. For vocab_size=128K
and seq_len=4K, this reduces peak memory from ~2GB to ~30MB.

Supports both:
- Non-quantized models: uses standard matmul with weight slicing
- Quantized models: uses mx.quantized_matmul with quantization-aware slicing

The key insight is to use online logsumexp to accumulate the normalization constant
incrementally while processing the vocabulary in chunks.
"""

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn


def _online_logsumexp_update(
    m: mx.array, s: mx.array, t: mx.array, chunk_logits: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Numerically stable online update for logsumexp and entropy accumulator.

    Maintains:
        m: running max (for numerical stability)
        s: running sum(exp(x - m))
        t: running sum(exp(x - m) * x) for entropy computation

    Args:
        m: Current running max, shape [N]
        s: Current running sum of exp, shape [N]
        t: Current running weighted sum for entropy, shape [N]
        chunk_logits: Logits for current chunk, shape [N, chunk_size]

    Returns:
        Updated (m, s, t) tuple
    """
    # Get max of current chunk
    chunk_m = mx.max(chunk_logits, axis=-1)  # [N]

    # New running max
    m_new = mx.maximum(m, chunk_m)  # [N]

    # Adjust previous accumulator for new max
    exp_adj = mx.exp(m - m_new)  # [N]

    # Compute chunk contributions with new max
    chunk_exp = mx.exp(chunk_logits - mx.expand_dims(m_new, axis=-1))  # [N, chunk_size]

    # Update accumulators (no in-place ops in MLX)
    s_new = s * exp_adj + mx.sum(chunk_exp, axis=-1)
    t_new = t * exp_adj + mx.sum(chunk_exp * chunk_logits, axis=-1)

    return m_new, s_new, t_new


def _gather_target_logits(
    logits_chunk: mx.array,
    labels: mx.array,
    target_logits: mx.array,
    start: int,
    end: int,
) -> mx.array:
    """
    Gather target logits for labels that fall within the current vocab chunk.

    Args:
        logits_chunk: Logits for current chunk, shape [N, chunk_size]
        labels: Target token indices, shape [N]
        target_logits: Accumulator for target logits, shape [N]
        start: Start index of this vocab chunk
        end: End index of this vocab chunk

    Returns:
        Updated target_logits with values gathered from this chunk
    """
    n = labels.shape[0]
    mask = (labels >= start) & (labels < end)

    if mx.any(mask):
        local_idx = labels - start
        safe_local_idx = mx.clip(local_idx, 0, end - start - 1)
        row_idx = mx.arange(n)
        gathered = logits_chunk[row_idx, safe_local_idx]
        target_logits = mx.where(mask, gathered, target_logits)

    return target_logits


# =============================================================================
# NON-QUANTIZED IMPLEMENTATION
# =============================================================================

def _chunked_logprobs_forward(
    hidden: mx.array,
    weight: mx.array,
    labels: mx.array,
    chunk_size: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Forward pass for chunked logprob computation (non-quantized weights).

    Args:
        hidden: Hidden states, shape [N, hidden_dim]
        weight: LM head weight matrix, shape [vocab_size, hidden_dim]
        labels: Target token indices, shape [N]
        chunk_size: Number of vocabulary tokens to process at once

    Returns:
        Tuple of (logprobs, entropy, logz)
    """
    n = hidden.shape[0]
    vocab_size = weight.shape[0]

    # Initialize running statistics in float32 for numerical stability
    m = mx.full((n,), float("-inf"), dtype=mx.float32)
    s = mx.zeros((n,), dtype=mx.float32)
    t = mx.zeros((n,), dtype=mx.float32)
    target_logits = mx.zeros((n,), dtype=mx.float32)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        w_chunk = weight[start:end]  # [chunk_size, hidden_dim]

        # Compute logits for this chunk
        logits_chunk = hidden @ w_chunk.T  # [N, chunk_size]
        logits_chunk = logits_chunk.astype(mx.float32)

        # Update running logsumexp and entropy accumulator
        m, s, t = _online_logsumexp_update(m, s, t, logits_chunk)

        # Capture target logits
        target_logits = _gather_target_logits(logits_chunk, labels, target_logits, start, end)

        # Evaluate periodically to prevent graph explosion
        mx.eval(m, s, t, target_logits)

    # Compute final values
    logz = m + mx.log(s)
    logprobs = target_logits - logz
    entropy = logz - (t / s)

    return logprobs, entropy, logz


def _chunked_logprobs_backward(
    grad_logprobs: mx.array,
    hidden: mx.array,
    weight: mx.array,
    labels: mx.array,
    logz: mx.array,
    chunk_size: int,
) -> mx.array:
    """
    Backward pass for chunked logprob computation (non-quantized weights).

    Args:
        grad_logprobs: Gradient w.r.t. logprobs, shape [N]
        hidden: Hidden states from forward pass, shape [N, hidden_dim]
        weight: LM head weight matrix (frozen), shape [vocab_size, hidden_dim]
        labels: Target token indices, shape [N]
        logz: Log partition function from forward pass, shape [N]
        chunk_size: Number of vocabulary tokens to process at once

    Returns:
        grad_hidden: Gradient w.r.t. hidden states, shape [N, hidden_dim]
    """
    n, hidden_dim = hidden.shape
    vocab_size = weight.shape[0]

    grad_hidden = mx.zeros((n, hidden_dim), dtype=hidden.dtype)
    g = grad_logprobs.astype(mx.float32)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        w_chunk = weight[start:end]

        # Recompute logits for this chunk
        logits_chunk = hidden @ w_chunk.T
        logits_chunk = logits_chunk.astype(mx.float32)

        # Compute softmax probabilities: p = exp(logits - logz)
        p = mx.exp(logits_chunk - mx.expand_dims(logz, axis=-1))

        # dL/d(logits) = g * (one_hot(label) - p)
        # Start with -g * p for all positions
        grad_logits = -mx.expand_dims(g, axis=-1) * p

        # Add +g for positions where label falls in this chunk
        mask = (labels >= start) & (labels < end)
        if mx.any(mask):
            local_idx = labels - start
            safe_local_idx = mx.clip(local_idx, 0, end - start - 1)
            row_idx = mx.arange(n)
            scatter_vals = mx.where(mask, g, mx.zeros_like(g))
            one_hot = mx.zeros_like(grad_logits)
            one_hot = one_hot.at[row_idx, safe_local_idx].add(scatter_vals)
            grad_logits = grad_logits + one_hot

        # Backprop through matmul: logits = hidden @ w_chunk.T
        # grad_hidden += grad_logits @ w_chunk
        grad_hidden = grad_hidden + (grad_logits.astype(hidden.dtype) @ w_chunk)

        mx.eval(grad_hidden)

    return grad_hidden


@mx.custom_function
def chunked_logprobs(
    hidden: mx.array,
    weight: mx.array,
    labels: mx.array,
    chunk_size: int,
) -> tuple[mx.array, mx.array]:
    """
    Compute log-probabilities and entropy using chunked processing (non-quantized).

    Args:
        hidden: Hidden states, shape [N, hidden_dim]
        weight: LM head weight matrix, shape [vocab_size, hidden_dim]
        labels: Target token indices, shape [N]
        chunk_size: Number of vocabulary tokens to process at once

    Returns:
        Tuple of (logprobs, entropy)
    """
    logprobs, entropy, _ = _chunked_logprobs_forward(hidden, weight, labels, chunk_size)
    return logprobs, entropy


@chunked_logprobs.vjp
def _chunked_logprobs_vjp(
    primals: tuple[mx.array, mx.array, mx.array, int],
    cotangents: tuple[mx.array, mx.array],
    outputs: tuple[mx.array, mx.array],
) -> tuple[mx.array, None, None, None]:
    """VJP for non-quantized chunked_logprobs."""
    hidden, weight, labels, chunk_size = primals
    grad_logprobs, grad_entropy = cotangents

    # Guard: entropy gradients are not supported
    if grad_entropy is not None:
        grad_entropy_sum = mx.sum(mx.abs(grad_entropy))
        mx.eval(grad_entropy_sum)
        if float(grad_entropy_sum) > 0:
            raise NotImplementedError(
                "Backward through entropy is not implemented. "
                "Use mx.stop_gradient(entropy) if needed."
            )

    # Recompute logz for backward pass
    _, _, logz = _chunked_logprobs_forward(hidden, weight, labels, chunk_size)

    grad_hidden = _chunked_logprobs_backward(
        grad_logprobs, hidden, weight, labels, logz, chunk_size
    )

    return (grad_hidden, None, None, None)


# =============================================================================
# QUANTIZED IMPLEMENTATION
# =============================================================================

def _chunked_logprobs_quantized_forward(
    hidden: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    labels: mx.array,
    chunk_size: int,
    group_size: int,
    bits: int,
    mode: str,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Forward pass for chunked logprob computation (quantized weights).

    Uses mx.quantized_matmul for memory-efficient computation with quantized weights.

    Args:
        hidden: Hidden states, shape [N, hidden_dim]
        weight: Quantized weight matrix, shape [vocab_size, packed_dim] (uint32)
        scales: Quantization scales, shape [vocab_size, num_groups]
        biases: Quantization biases, shape [vocab_size, num_groups] or None
        labels: Target token indices, shape [N]
        chunk_size: Number of vocabulary tokens to process at once
        group_size: Quantization group size
        bits: Quantization bits (e.g., 4 or 8)
        mode: Quantization mode (e.g., "affine")

    Returns:
        Tuple of (logprobs, entropy, logz)
    """
    n = hidden.shape[0]
    vocab_size = weight.shape[0]

    # Initialize running statistics in float32 for numerical stability
    m = mx.full((n,), float("-inf"), dtype=mx.float32)
    s = mx.zeros((n,), dtype=mx.float32)
    t = mx.zeros((n,), dtype=mx.float32)
    target_logits = mx.zeros((n,), dtype=mx.float32)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)

        # Slice quantized weights and their parameters
        w_chunk = weight[start:end]
        s_chunk = scales[start:end]
        b_chunk = biases[start:end] if biases is not None else None

        # Compute logits using quantized matmul
        # This dequantizes on-the-fly: hidden @ dequant(w_chunk).T
        logits_chunk = mx.quantized_matmul(
            hidden,
            w_chunk,
            scales=s_chunk,
            biases=b_chunk,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        logits_chunk = logits_chunk.astype(mx.float32)

        # Update running logsumexp and entropy accumulator
        m, s, t = _online_logsumexp_update(m, s, t, logits_chunk)

        # Capture target logits
        target_logits = _gather_target_logits(logits_chunk, labels, target_logits, start, end)

        # Evaluate periodically to prevent graph explosion
        mx.eval(m, s, t, target_logits)

    # Compute final values
    logz = m + mx.log(s)
    logprobs = target_logits - logz
    entropy = logz - (t / s)

    return logprobs, entropy, logz


def _chunked_logprobs_quantized_backward(
    grad_logprobs: mx.array,
    hidden: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    labels: mx.array,
    logz: mx.array,
    chunk_size: int,
    group_size: int,
    bits: int,
    mode: str,
) -> mx.array:
    """
    Backward pass for chunked logprob computation (quantized weights).

    The gradient formula is:
        d_logprobs/d_hidden = (one_hot(label) - softmax(logits)) @ W

    where W is the dequantized weight matrix. We compute this in chunks using
    mx.quantized_matmul with transpose=False.

    Args:
        grad_logprobs: Gradient w.r.t. logprobs, shape [N]
        hidden: Hidden states from forward pass, shape [N, hidden_dim]
        weight: Quantized weight matrix, shape [vocab_size, packed_dim]
        scales: Quantization scales, shape [vocab_size, num_groups]
        biases: Quantization biases or None
        labels: Target token indices, shape [N]
        logz: Log partition function from forward pass, shape [N]
        chunk_size: Vocabulary chunk size
        group_size: Quantization group size
        bits: Quantization bits
        mode: Quantization mode

    Returns:
        grad_hidden: Gradient w.r.t. hidden states, shape [N, hidden_dim]
    """
    n = hidden.shape[0]
    vocab_size = weight.shape[0]

    # Infer hidden_dim from the quantization parameters
    # For 8-bit with group_size=64: packed_dim = hidden_dim / 4
    # scales shape is [vocab, num_groups] where num_groups = hidden_dim / group_size
    num_groups = scales.shape[1]
    hidden_dim = num_groups * group_size

    grad_hidden = mx.zeros((n, hidden_dim), dtype=hidden.dtype)
    g = grad_logprobs.astype(mx.float32)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)

        # Slice quantized weights
        w_chunk = weight[start:end]
        s_chunk = scales[start:end]
        b_chunk = biases[start:end] if biases is not None else None

        # Recompute logits for this chunk using quantized matmul
        logits_chunk = mx.quantized_matmul(
            hidden,
            w_chunk,
            scales=s_chunk,
            biases=b_chunk,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        logits_chunk = logits_chunk.astype(mx.float32)

        # Compute softmax probabilities: p = exp(logits - logz)
        p = mx.exp(logits_chunk - mx.expand_dims(logz, axis=-1))  # [N, chunk_size]

        # dL/d(logits) = g * (one_hot(label) - p)
        # Start with -g * p for all positions
        grad_logits = -mx.expand_dims(g, axis=-1) * p  # [N, chunk_size]

        # Add +g for positions where label falls in this chunk
        mask = (labels >= start) & (labels < end)
        if mx.any(mask):
            local_idx = labels - start
            safe_local_idx = mx.clip(local_idx, 0, end - start - 1)
            row_idx = mx.arange(n)
            scatter_vals = mx.where(mask, g, mx.zeros_like(g))
            one_hot = mx.zeros_like(grad_logits)
            one_hot = one_hot.at[row_idx, safe_local_idx].add(scatter_vals)
            grad_logits = grad_logits + one_hot

        # Backprop through quantized matmul: logits = hidden @ W.T
        # grad_hidden += grad_logits @ W
        # With quantized: grad_logits @ dequant(W) using transpose=False
        grad_chunk = mx.quantized_matmul(
            grad_logits.astype(hidden.dtype),
            w_chunk,
            scales=s_chunk,
            biases=b_chunk,
            transpose=False,  # Computes grad_logits @ dequant(W)
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        grad_hidden = grad_hidden + grad_chunk

        mx.eval(grad_hidden)

    return grad_hidden


@mx.custom_function
def chunked_logprobs_quantized(
    hidden: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,  # Can be zeros array if no biases
    labels: mx.array,
    chunk_size: int,
    group_size: int,
    bits: int,
) -> tuple[mx.array, mx.array]:
    """
    Compute log-probabilities and entropy using chunked processing (quantized).

    This is the main entry point for quantized models, decorated with @mx.custom_function
    to provide a custom VJP for gradient computation.

    Args:
        hidden: Hidden states, shape [N, hidden_dim]
        weight: Quantized weight matrix, shape [vocab_size, packed_dim]
        scales: Quantization scales, shape [vocab_size, num_groups]
        biases: Quantization biases, shape [vocab_size, num_groups] (pass zeros if none)
        labels: Target token indices, shape [N]
        chunk_size: Vocabulary chunk size
        group_size: Quantization group size
        bits: Quantization bits

    Returns:
        Tuple of (logprobs, entropy)
    """
    # Handle None biases by checking if it's a zeros array with wrong shape
    actual_biases = biases if biases.shape == scales.shape else None

    logprobs, entropy, _ = _chunked_logprobs_quantized_forward(
        hidden, weight, scales, actual_biases, labels, chunk_size, group_size, bits, "affine"
    )
    return logprobs, entropy


@chunked_logprobs_quantized.vjp
def _chunked_logprobs_quantized_vjp(
    primals: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, int, int, int],
    cotangents: tuple[mx.array, mx.array],
    outputs: tuple[mx.array, mx.array],
) -> tuple[mx.array, None, None, None, None, None, None, None]:
    """VJP for quantized chunked_logprobs."""
    hidden, weight, scales, biases, labels, chunk_size, group_size, bits = primals
    grad_logprobs, grad_entropy = cotangents

    # Guard: entropy gradients are not supported
    if grad_entropy is not None:
        grad_entropy_sum = mx.sum(mx.abs(grad_entropy))
        mx.eval(grad_entropy_sum)
        if float(grad_entropy_sum) > 0:
            raise NotImplementedError(
                "Backward through entropy is not implemented. "
                "Use mx.stop_gradient(entropy) if needed."
            )

    # Handle None biases
    actual_biases = biases if biases.shape == scales.shape else None

    # Recompute logz for backward pass
    _, _, logz = _chunked_logprobs_quantized_forward(
        hidden, weight, scales, actual_biases, labels, chunk_size, group_size, bits, "affine"
    )

    grad_hidden = _chunked_logprobs_quantized_backward(
        grad_logprobs, hidden, weight, scales, actual_biases, labels, logz,
        chunk_size, group_size, bits, "affine"
    )

    # Return None for all non-differentiable inputs
    return (grad_hidden, None, None, None, None, None, None, None)


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def compute_chunked_logprobs(
    hidden: mx.array,
    embed_or_weight: Union[nn.Embedding, nn.QuantizedEmbedding, mx.array],
    labels: mx.array,
    chunk_size: int,
) -> tuple[mx.array, mx.array]:
    """
    Compute log-probabilities and entropy using chunked processing.

    Automatically detects whether to use quantized or non-quantized path based
    on the type of embed_or_weight.

    Args:
        hidden: Hidden states, shape [N, hidden_dim]
        embed_or_weight: Either:
            - nn.QuantizedEmbedding: Uses quantized matmul path
            - nn.Embedding: Uses non-quantized path with embedding weight
            - mx.array: Uses non-quantized path directly with weight matrix
        labels: Target token indices, shape [N]
        chunk_size: Number of vocabulary tokens to process at once

    Returns:
        Tuple of (logprobs, entropy) where:
            - logprobs: Log probabilities of target tokens, shape [N]
            - entropy: Entropy of the distribution, shape [N]
    """
    if isinstance(embed_or_weight, nn.QuantizedEmbedding):
        # Quantized path
        embed = embed_or_weight
        weight = embed.weight
        scales = embed.scales
        # Handle missing biases by passing a dummy array with wrong shape
        biases = embed.biases if embed.biases is not None else mx.zeros((1,))
        group_size = embed.group_size
        bits = embed.bits

        return chunked_logprobs_quantized(
            hidden, weight, scales, biases, labels,
            chunk_size, group_size, bits
        )

    elif isinstance(embed_or_weight, nn.Embedding):
        # Non-quantized embedding
        return chunked_logprobs(hidden, embed_or_weight.weight, labels, chunk_size)

    else:
        # Assume it's a raw weight matrix
        return chunked_logprobs(hidden, embed_or_weight, labels, chunk_size)


def is_quantized_embedding(embed: Union[nn.Module, mx.array]) -> bool:
    """Check if an embedding layer is quantized."""
    return isinstance(embed, nn.QuantizedEmbedding)


# =============================================================================
# LEGACY CLASS INTERFACE (for backwards compatibility)
# =============================================================================

class FusedLMHead(nn.Module):
    """
    Memory-efficient LM head that computes logprobs and entropy without
    materializing the full vocabulary tensor.

    This module wraps the chunked_logprobs function for use in MLX models.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, chunk_size: int = 2048):
        super().__init__()
        self.weight = mx.zeros((vocab_size, hidden_dim))
        self.chunk_size = chunk_size

    def __call__(self, hidden: mx.array, labels: mx.array) -> tuple[mx.array, mx.array]:
        return chunked_logprobs(hidden, self.weight, labels, self.chunk_size)

    @classmethod
    def from_linear(cls, linear: nn.Linear, chunk_size: int = 2048) -> "FusedLMHead":
        vocab_size, hidden_dim = linear.weight.shape
        head = cls(hidden_dim, vocab_size, chunk_size)
        head.weight = linear.weight
        return head
