#!/usr/bin/env python3
"""
vLLM Wrapper for LlamaIndex Integration
Provides GPU-accelerated LLM inference with vLLM backend
"""
import os
from typing import Optional
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from vllm import LLM, SamplingParams


class vLLMWrapper(CustomLLM):
    """
    LlamaIndex-compatible wrapper for vLLM.

    vLLM provides GPU-optimized inference with:
    - Automatic GPU detection and usage
    - PagedAttention for efficient memory
    - Continuous batching
    - Flash Attention support
    - 10-20x faster than llama.cpp CPU

    Usage:
        llm = vLLMWrapper(
            model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            temperature=0.1,
            max_tokens=256,
        )
    """

    model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    temperature: float = 0.1
    max_tokens: int = 256
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096

    _llm: Optional[LLM] = None

    def __init__(
        self,
        model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        temperature: float = 0.1,
        max_tokens: int = 256,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        # Initialize vLLM engine (lazy loading)
        self._initialize()

    def _initialize(self):
        """Initialize vLLM engine with GPU support"""
        if self._llm is None:
            print(f"🚀 Initializing vLLM with {self.model_name}...")
            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=False,
                dtype="float16",
            )
            print(f"✅ vLLM initialized on GPU")

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata"""
        return LLMMetadata(
            context_window=self.max_model_len,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Generate completion"""
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95,
        )

        outputs = self._llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text

        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        """Streaming not implemented for vLLM wrapper"""
        # vLLM supports streaming but requires async
        # For simplicity, fall back to complete()
        response = self.complete(prompt, **kwargs)
        yield response


def build_vllm_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 256,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 4096,
) -> vLLMWrapper:
    """
    Build vLLM-based LLM for LlamaIndex.

    Args:
        model_name: HuggingFace model name (AWQ quantized recommended)
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        max_model_len: Maximum context length

    Returns:
        vLLMWrapper instance ready for use with LlamaIndex

    Example:
        llm = build_vllm_llm(
            model_name="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            temperature=0.1,
            max_tokens=256,
        )
    """
    # Default model from env or use AWQ Mistral
    if model_name is None:
        model_name = os.getenv(
            "VLLM_MODEL",
            "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
        )

    return vLLMWrapper(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )


if __name__ == "__main__":
    # Test the wrapper
    print("Testing vLLM wrapper...")
    llm = build_vllm_llm(temperature=0.1, max_tokens=50)

    response = llm.complete("What is the capital of France?")
    print(f"✅ Response: {response.text}")
