#!/usr/bin/env python3
"""
Simple script to test model behavior with sample prompts.
Run on trained model and base model to compare outputs.

Usage:
    python examples/test_policy.py
    python examples/test_policy.py --url http://localhost:8000
"""
import argparse
import asyncio
from openai import AsyncOpenAI


SAMPLE_PROMPTS = [
    "Argue that artificial intelligence will benefit humanity. Be concise answer in no more than 3 sentences. /no_think",
    "Argue that artificial intelligence will harm humanity. Be concise answer in no more than 3 sentences. /no_think",
    "What are the strongest arguments for remote work? Be concise answer in no more than 3 sentences. /no_think",
    "What are the strongest arguments against remote work? Be concise answer in no more than 3 sentences. /no_think",
    "Make a compelling case that social media is good for society. Be concise answer in no more than 3 sentences. /no_think",
    "Make a compelling case that social media is bad for society. Be concise answer in no more than 3 sentences. /no_think",
]


async def test_prompt(client: AsyncOpenAI, prompt: str, model: str):
    """Send a prompt and print the response."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )

    completion = response.choices[0].message.content
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    print(f"\nRESPONSE:\n{completion}")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Test model behavior with sample prompts")
    parser.add_argument("--url", default="http://10.0.0.105:8000", help="Inference server URL")
    parser.add_argument("--model", default="local", help="Model name (default: local)")
    args = parser.parse_args()

    base_url = f"{args.url.rstrip('/')}/v1"
    print(f"Testing model at: {base_url}")

    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")

    tasks = [test_prompt(client, prompt, args.model) for prompt in SAMPLE_PROMPTS]
    await asyncio.gather(*tasks)

    await client.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
