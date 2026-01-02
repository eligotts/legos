"""
Weight publisher for pushing LoRA weights to inference server.

Handles serialization of MLX model weights to safetensors format,
base64 encoding, and HTTP POST to the mlx-vllm /adapters/load endpoint.
"""
from __future__ import annotations

import base64
import time
from typing import Optional

import httpx
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from safetensors.numpy import save as save_safetensors


class WeightPublisher:
    """
    Publishes LoRA adapter weights to an mlx-vllm inference server.

    After each training step, the updated LoRA weights are serialized and
    pushed to the inference server, enabling online learning with immediate
    effect on generation.

    The protocol:
    1. Extract trainable parameters (LoRA weights) from model
    2. Convert MLX arrays to numpy
    3. Serialize to safetensors format
    4. Base64 encode
    5. POST to /adapters/load with version number

    The version number is typically the training step, allowing the inference
    server to track staleness and the Arena to tag generated data with the
    policy version that produced it.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        """
        Initialize the weight publisher.

        Args:
            base_url: Base URL of the mlx-vllm server (without /v1)
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def publish(
        self,
        model,
        version: int,
        verbose: bool = False,
    ) -> dict:
        """
        Push updated LoRA weights to the inference server.

        Args:
            model: MLX model with trainable LoRA parameters
            version: Version number (typically training step)
            verbose: Print timing info

        Returns:
            Response dict from server with {"status": "ok", "version": N}
        """
        timings = {}

        # 1. Extract LoRA weights (trainable parameters only)
        t0 = time.perf_counter()
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        timings["extract"] = time.perf_counter() - t0

        if not adapter_weights:
            raise ValueError("Model has no trainable parameters. Did you attach LoRA?")

        # 2. Evaluate to materialize lazy arrays
        t0 = time.perf_counter()
        mx.eval(adapter_weights)
        timings["eval"] = time.perf_counter() - t0

        # 3. Convert to numpy (cast bfloat16 to float32 since numpy doesn't support bf16)
        t0 = time.perf_counter()
        def to_numpy(arr: mx.array) -> np.ndarray:
            if arr.dtype == mx.bfloat16:
                return np.array(arr.astype(mx.float32))
            return np.array(arr)

        weights_np = {k: to_numpy(v) for k, v in adapter_weights.items()}
        timings["to_numpy"] = time.perf_counter() - t0

        # 4. Serialize to safetensors
        t0 = time.perf_counter()
        weight_bytes = save_safetensors(weights_np)
        timings["safetensors"] = time.perf_counter() - t0

        # 5. Base64 encode
        t0 = time.perf_counter()
        weights_b64 = base64.b64encode(weight_bytes).decode("utf-8")
        timings["base64"] = time.perf_counter() - t0

        if verbose:
            total_bytes = len(weight_bytes)
            print(f"      [publish] {len(adapter_weights)} tensors, {total_bytes/1024:.1f}KB")
            print(f"      [publish] extract={timings['extract']*1000:.1f}ms, "
                  f"eval={timings['eval']*1000:.1f}ms, "
                  f"numpy={timings['to_numpy']*1000:.1f}ms, "
                  f"safetensors={timings['safetensors']*1000:.1f}ms, "
                  f"base64={timings['base64']*1000:.1f}ms")

        # 6. POST to /adapters/load
        t0 = time.perf_counter()
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/adapters/load",
            json={"weights": weights_b64, "version": version},
        )
        response.raise_for_status()
        timings["http_post"] = time.perf_counter() - t0

        if verbose:
            print(f"      [publish] http_post={timings['http_post']*1000:.1f}ms")

        return response.json()

    async def get_version(self) -> int:
        """
        Get the current adapter version from the inference server.

        Returns:
            Current version number, or 0 if not available
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/adapters/version",
                timeout=5.0,
            )
            if response.status_code == 200:
                return response.json().get("version", 0)
            return 0
        except Exception:
            return 0

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
