"""
Model abstraction layer for RepoAI experiments.
Replaces all hardcoded model references with configurable providers.
Supports NVIDIA NIM (OpenAI-compatible) and Google Gemini (Vertex AI / AI Studio).
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for all models used in the experiment pipeline."""

    # Generator (code generation)
    generator_model: str = "qwen/qwen2.5-coder-32b-instruct"
    generator_provider: str = "nim"  # "nim" | "gemini"
    generator_temperature: float = 0.3
    generator_max_tokens: int = 24000

    # Router (intent classification)
    router_model: str = "qwen/qwen2.5-coder-32b-instruct"
    router_provider: str = "nim"
    router_temperature: float = 0.2

    # Judge (intent verification -- must differ from generator family)
    # Default: Llama 3.3 70B on NIM (cross-family from Qwen generator, unlimited free tier)
    judge_model: str = "meta/llama-3.3-70b-instruct"
    judge_provider: str = "nim"
    judge_temperature: float = 0.2

    # Embeddings
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    embedding_provider: str = "nim"  # "nim" | "gemini"

    # NIM settings
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_api_keys: list = field(default_factory=list)
    _nim_key_index: int = field(default=0, repr=False)

    # Gemini settings
    gemini_api_key: str = ""

    def __post_init__(self):
        # Load from env if not explicitly set
        if not self.nim_api_keys:
            keys = []
            for var in ["NVIDIA_API_KEY", "NVIDIA_API_KEY_2", "NVIDIA_API_KEY_3"]:
                k = os.environ.get(var, "")
                if k:
                    keys.append(k)
            self.nim_api_keys = keys if keys else [""]

        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")

    @property
    def nim_api_key(self) -> str:
        """Round-robin across NIM API keys to distribute rate limits."""
        key = self.nim_api_keys[self._nim_key_index % len(self.nim_api_keys)]
        self._nim_key_index += 1
        return key


# --- Predefined experiment configurations ---

CONFIGS = {
    # EXP-1 conditions (all use NIM Qwen for generation)
    "c1_direct": ModelConfig(
        generator_model="qwen/qwen2.5-coder-32b-instruct",
        generator_provider="nim",
    ),
    "c2_rag": ModelConfig(
        generator_model="qwen/qwen2.5-coder-32b-instruct",
        generator_provider="nim",
    ),
    "c3_multiagent": ModelConfig(
        generator_model="qwen/qwen2.5-coder-32b-instruct",
        generator_provider="nim",
    ),
    "c4_full": ModelConfig(
        generator_model="qwen/qwen2.5-coder-32b-instruct",
        generator_provider="nim",
    ),
    "c5_memory": ModelConfig(
        generator_model="qwen/qwen2.5-coder-32b-instruct",
        generator_provider="nim",
    ),

    # EXP-2 multi-model (cross-family judge: always Llama 3.3 70B)
    "nim_qwen": ModelConfig(
        generator_model="qwen/qwen2.5-coder-32b-instruct",
        generator_provider="nim",
        judge_model="meta/llama-3.3-70b-instruct",
        judge_provider="nim",
    ),
    "nim_deepseek": ModelConfig(
        generator_model="deepseek-ai/deepseek-r1-distill-qwen-32b",
        generator_provider="nim",
        judge_model="meta/llama-3.3-70b-instruct",
        judge_provider="nim",
    ),
    "nim_devstral": ModelConfig(
        generator_model="mistralai/devstral-2-123b-instruct-2512",
        generator_provider="nim",
        judge_model="meta/llama-3.3-70b-instruct",
        judge_provider="nim",
    ),
    "gemini_flash": ModelConfig(
        generator_model="gemini-2.5-flash",
        generator_provider="gemini",
        judge_model="qwen/qwen2.5-coder-32b-instruct",
        judge_provider="nim",
    ),
    "gemini_pro": ModelConfig(
        generator_model="gemini-2.5-pro",
        generator_provider="gemini",
        judge_model="qwen/qwen2.5-coder-32b-instruct",
        judge_provider="nim",
    ),

    # EXP-4 ablation control
    "a5_ctrl": ModelConfig(
        generator_model="qwen/qwen2.5-coder-32b-instruct",
        generator_provider="nim",
    ),

    # EXP-6 RAG embedding configs
    "rag_gemini_embed": ModelConfig(
        embedding_model="models/gemini-embedding-001",
        embedding_provider="gemini",
    ),
    "rag_nemotron_embed": ModelConfig(
        embedding_model="nvidia/llama-nemotron-embed-1b-v2",
        embedding_provider="nim",
    ),
    "rag_bge_embed": ModelConfig(
        embedding_model="baai/bge-m3",
        embedding_provider="nim",
    ),

    # Llama models (cross-family judge: Qwen)
    "nim_llama33": ModelConfig(
        generator_model="meta/llama-3.3-70b-instruct",
        generator_provider="nim",
        judge_model="qwen/qwen2.5-coder-32b-instruct",
        judge_provider="nim",
    ),
    "nim_llama4_maverick": ModelConfig(
        generator_model="meta/llama-4-maverick-17b-128e-instruct",
        generator_provider="nim",
        judge_model="qwen/qwen2.5-coder-32b-instruct",
        judge_provider="nim",
    ),

    # Google Gemma 2 9B (small model baseline, cross-family judge: Qwen)
    "nim_gemma2_9b": ModelConfig(
        generator_model="google/gemma-2-9b-it",
        generator_provider="nim",
        judge_model="qwen/qwen2.5-coder-32b-instruct",
        judge_provider="nim",
    ),
}


class _NIMChatWrapper:
    """Thin wrapper around OpenAI client for NIM, compatible with timed_llm_call."""

    def __init__(self, client, model, temperature, max_tokens):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, prompt):
        """Call NIM API and return a response object with .content and .usage_metadata."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # LangChain message format
            messages = []
            for m in prompt:
                if hasattr(m, 'content'):
                    role = getattr(m, 'type', 'user')
                    if role == 'human':
                        role = 'user'
                    elif role == 'ai':
                        role = 'assistant'
                    elif role == 'system':
                        role = 'system'
                    messages.append({"role": role, "content": m.content})
                else:
                    messages.append(m)
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Return an object with .content and .usage_metadata like LangChain
        return _NIMResponse(resp)


class _NIMResponse:
    """Mimics LangChain AIMessage for compatibility with timed_llm_call."""

    def __init__(self, openai_response):
        self.content = openai_response.choices[0].message.content or ""
        usage = openai_response.usage
        self.usage_metadata = None
        self.response_metadata = {}
        if usage:
            self.response_metadata["token_usage"] = {
                "prompt_tokens": usage.prompt_tokens or 0,
                "completion_tokens": usage.completion_tokens or 0,
            }


def get_llm(config: ModelConfig, role: str = "generator"):
    """
    Factory: returns a LangChain Chat LLM for the given role.
    role: "generator" | "router" | "judge"
    """
    if role == "generator":
        model, provider, temp, max_tok = (
            config.generator_model, config.generator_provider,
            config.generator_temperature, config.generator_max_tokens,
        )
    elif role == "router":
        model, provider, temp, max_tok = (
            config.router_model, config.router_provider,
            config.router_temperature, 4000,
        )
    elif role == "judge":
        model, provider, temp, max_tok = (
            config.judge_model, config.judge_provider,
            config.judge_temperature, 4000,
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    if provider == "nim":
        from openai import OpenAI
        # NIM doesn't support max_completion_tokens. Use raw OpenAI client
        # wrapped in a thin LangChain-compatible adapter.
        client = OpenAI(base_url=config.nim_base_url, api_key=config.nim_api_key)
        return _NIMChatWrapper(client, model, temp, max_tok)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            max_output_tokens=max_tok,
            google_api_key=config.gemini_api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embeddings(config: ModelConfig):
    """Factory: returns a LangChain Embeddings model."""
    if config.embedding_provider == "nim":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.embedding_model,
            base_url=config.nim_base_url,
            api_key=config.nim_api_key,
        )
    elif config.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.gemini_api_key,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")


# --- Instrumentation ---

class TokenTracker:
    """Tracks token usage and timing across all LLM calls for a task."""

    def __init__(self, task_id: str, experiment: str, condition: str):
        self.task_id = task_id
        self.experiment = experiment
        self.condition = condition
        self.calls = []

    def record(self, stage: str, model: str, provider: str,
               input_tokens: int, output_tokens: int, latency_ms: float,
               success: bool = True, error: str = ""):
        self.calls.append({
            "stage": stage,
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
        })

    @property
    def total_input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)

    @property
    def total_latency_ms(self) -> float:
        return sum(c["latency_ms"] for c in self.calls)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "experiment": self.experiment,
            "condition": self.condition,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_latency_ms": self.total_latency_ms,
            "num_calls": len(self.calls),
            "calls": self.calls,
        }


def timed_llm_call(llm, prompt, tracker: TokenTracker, stage: str,
                   model: str, provider: str, max_retries: int = 3):
    """
    Call an LLM with retry logic and token tracking.
    Returns the response content string.
    """
    for attempt in range(max_retries):
        start = time.time()
        try:
            response = llm.invoke(prompt)
            latency = (time.time() - start) * 1000

            # Extract token counts
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                meta = response.usage_metadata
                input_tokens = getattr(meta, 'input_tokens', 0) or 0
                output_tokens = getattr(meta, 'output_tokens', 0) or 0
            elif hasattr(response, 'response_metadata'):
                meta = response.response_metadata
                usage = meta.get('token_usage', meta.get('usage', {}))
                if isinstance(usage, dict):
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)

            tracker.record(stage, model, provider,
                           input_tokens, output_tokens, latency)

            content = response.content if hasattr(response, 'content') else str(response)
            return content

        except Exception as e:
            latency = (time.time() - start) * 1000
            error_str = str(e)
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {error_str}")

            if attempt == max_retries - 1:
                tracker.record(stage, model, provider, 0, 0, latency,
                               success=False, error=error_str)
                raise

            # Exponential backoff
            wait = min(2 ** attempt * 5, 60)
            logger.info(f"Retrying in {wait}s...")
            time.sleep(wait)
