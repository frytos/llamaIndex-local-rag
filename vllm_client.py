#!/usr/bin/env python3
"""
vLLM Client for LlamaIndex Integration
Uses vLLM OpenAI-compatible API server for ultra-fast queries
"""
import os
from typing import Optional
from llama_index.llms.openai import OpenAI


def build_vllm_client(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> OpenAI:
    """
    Build OpenAI-compatible client for vLLM server.

    Prerequisites:
        vLLM server must be running:
        $ vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ --port 8000

    Args:
        base_url: vLLM server URL (default: http://localhost:8000/v1)
        model: Model name (must match server)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        OpenAI client configured for vLLM server

    Performance:
        First query: ~60s (server warmup)
        Subsequent queries: ~2-5s (no reload!)

    Example:
        # Start server first:
        $ vllm serve TheBloke/Mistral-7B-Instruct-v0.2-AWQ --port 8000

        # Then use client:
        llm = build_vllm_client()
        response = llm.complete("test query")
    """
    if base_url is None:
        vllm_port = os.getenv("VLLM_PORT", "8000")
        base_url = f"http://localhost:{vllm_port}/v1"

    if model is None:
        model = os.getenv("VLLM_MODEL", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")

    # Test connection
    import requests
    try:
        health_url = base_url.replace("/v1", "/health")
        resp = requests.get(health_url, timeout=2)
        if resp.status_code != 200:
            raise ConnectionError(f"vLLM server not healthy: {resp.status_code}")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"❌ Cannot connect to vLLM server at {base_url}\n"
            f"   Make sure vLLM server is running:\n"
            f"   $ ./scripts/start_vllm_server.sh\n"
            f"   Original error: {e}"
        )

    # Create OpenAI client pointing to vLLM
    # Use generic OpenAI model name to bypass LlamaIndex validation
    # The vLLM server will use its own loaded model regardless
    # Note: The 'model' param is just for validation, vLLM uses its loaded model
    llm = OpenAI(
        api_base=base_url,
        api_key="dummy",  # vLLM doesn't check
        model="gpt-3.5-turbo",  # Generic name for LlamaIndex validation
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Log actual model for reference
    print(f"  💡 vLLM server loaded model: {model}")
    print(f"  🔗 Connected to: {base_url}")

    return llm


if __name__ == "__main__":
    # Test the client
    print("Testing vLLM client...")

    try:
        llm = build_vllm_client(temperature=0.1, max_tokens=50)
        print("✅ Connected to vLLM server")

        response = llm.complete("What is 2+2?")
        print(f"✅ Response: {response.text}")
    except ConnectionError as e:
        print(str(e))
