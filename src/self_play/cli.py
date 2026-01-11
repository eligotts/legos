"""
CLI for the self-play engine.

Usage:
    self-play serve [OPTIONS]     Start the inference server
    self-play --help              Show help

Examples:
    self-play serve
    self-play serve --model /path/to/model
    self-play serve --port 8080

LoRA settings are configured in src/self_play/lora.py
"""

import argparse
import sys


def serve_command(args: argparse.Namespace) -> None:
    """Start the inference server."""
    import uvicorn

    from self_play.inference.config import ServerConfig
    from self_play.inference.server import set_config

    # Build config, overriding with CLI args if provided
    config_overrides = {}
    if args.model:
        config_overrides["model_path"] = args.model
    if args.host:
        config_overrides["host"] = args.host
    if args.port:
        config_overrides["port"] = args.port
    if args.max_batch_size:
        config_overrides["max_batch_size"] = args.max_batch_size
    if args.max_tokens:
        config_overrides["max_tokens"] = args.max_tokens
    # Sampler args
    if args.temperature is not None:
        config_overrides["temperature"] = args.temperature
    if args.top_p is not None:
        config_overrides["top_p"] = args.top_p
    if args.top_k is not None:
        config_overrides["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        config_overrides["repetition_penalty"] = args.repetition_penalty

    config = ServerConfig(**config_overrides)
    set_config(config)

    print("Starting Self-Play Inference Server")
    print(f"  Model: {config.model_path}")
    print(f"  Host: {config.host}:{config.port}")
    print(f"  Max batch size: {config.max_batch_size}")
    print(f"  Default max tokens: {config.max_tokens}")
    print(f"  Sampler: temp={config.temperature}, top_p={config.top_p}, top_k={config.top_k}, rep_penalty={config.repetition_penalty}")
    print(f"  LoRA: {'enabled' if config.enable_lora else 'disabled'}")

    uvicorn.run(
        "self_play.inference.server:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


def main() -> None:
    """Main entry point for the self-play CLI."""
    parser = argparse.ArgumentParser(
        prog="self-play",
        description="Self-Play LLM RL Engine - Training and inference for self-play scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  self-play serve                      Start inference server
  self-play serve --model /path/to/m   Use custom model
  self-play serve --port 8080          Use custom port

LoRA settings: edit src/self_play/lora.py
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the inference server",
        description="Start an OpenAI-compatible inference server with optional LoRA support",
    )
    serve_parser.add_argument(
        "--model",
        type=str,
        help="Model path or HuggingFace repo ID",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for continuous batching (default: 32)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Default max tokens per request (default: 4096)",
    )
    # Sampler arguments
    serve_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.7)",
    )
    serve_parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling (default: 1.0)",
    )
    serve_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling, -1 to disable (default: -1)",
    )
    serve_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty, 1.0 = none, >1.0 penalizes (default: 1.0)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        serve_command(args)


if __name__ == "__main__":
    main()
