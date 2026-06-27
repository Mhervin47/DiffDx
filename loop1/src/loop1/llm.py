from __future__ import annotations

import json
import os
import re
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from loop1.config import config

_GROQ_BASE = "https://api.groq.com/openai/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_CEREBRAS_BASE = "https://api.cerebras.ai/v1"
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_RETRYABLE_STATUS = {500, 502, 503, 504}  # 429 handled separately via fallback

# When a provider hits 429, try these fallbacks in order
_FALLBACK_CHAIN: dict[str, list[str]] = {
    "groq": [
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "openrouter/google/gemma-4-31b-it:free",
        "openrouter/deepseek/deepseek-r1:free",
        "openrouter/qwen/qwen3-30b-a3b:free",
        "openrouter/mistralai/mistral-7b-instruct:free",
        "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    ],
    "gemini": [
        "groq/llama-3.3-70b-versatile",
        "openrouter/google/gemma-4-31b-it:free",
        "openrouter/deepseek/deepseek-r1:free",
        "openrouter/qwen/qwen3-30b-a3b:free",
    ],
    "openrouter": [
        "groq/llama-3.3-70b-versatile",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "openrouter/deepseek/deepseek-r1:free",
        "openrouter/qwen/qwen3-30b-a3b:free",
        "openrouter/mistralai/mistral-7b-instruct:free",
        "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    ],
    "cerebras": [
        "groq/llama-3.3-70b-versatile",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    ],
}


def _groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")
    return key


def _openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return key


def _cerebras_api_key() -> str:
    key = os.environ.get("CEREBRAS_API_KEY", "")
    if not key:
        raise RuntimeError("CEREBRAS_API_KEY environment variable not set")
    return key


def _gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    return key


def _is_retryable(exc: BaseException) -> bool:
    import httpx
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, _RetryableHTTPError))


class _RetryableHTTPError(Exception):
    pass


class _RateLimitError(Exception):
    """Raised on HTTP 429 — triggers provider fallback, not same-provider retry."""
    pass


def _provider_prefix(model: str) -> str:
    if "/" in model:
        return model.split("/")[0]
    return "groq"  # default


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _call_llm_raw(model: str, messages: list[dict[str, str]], **kwargs: Any) -> tuple[str, int]:
    import httpx  # lazy import — avoids SSL/Keychain hang at server startup on macOS
    defaults: dict[str, Any] = {
        "temperature": config["thresholds"]["llm_temperature"],
        "max_tokens": config["thresholds"]["llm_max_tokens"],
    }
    defaults.update(kwargs)

    if model.startswith("gemini/"):
        base_url = _GEMINI_BASE
        api_key = _gemini_api_key()
        model_name = model.removeprefix("gemini/")
    elif model.startswith("openrouter/"):
        base_url = _OPENROUTER_BASE
        api_key = _openrouter_api_key()
        model_name = model.removeprefix("openrouter/")
    elif model.startswith("cerebras/"):
        key = os.environ.get("CEREBRAS_API_KEY", "")
        if not key:
            raise _RateLimitError("CEREBRAS_API_KEY not set — skipping cerebras model")
        base_url = _CEREBRAS_BASE
        api_key = key
        model_name = model.removeprefix("cerebras/")
        # Reasoning models burn tokens on CoT before output — require generous headroom
        defaults.setdefault("max_tokens", 8192)
        if defaults.get("max_tokens", 0) < 8192:
            defaults["max_tokens"] = 8192
    else:
        base_url = _GROQ_BASE
        api_key = _groq_api_key()
        model_name = model.removeprefix("groq/")

    payload = {
        "model": model_name,
        "messages": messages,
        **defaults,
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code == 429:
        raise _RateLimitError(f"HTTP 429: {response.text[:300]}")
    if response.status_code in _RETRYABLE_STATUS:
        raise _RetryableHTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text[:400]}")

    data = response.json()
    msg = data["choices"][0]["message"]
    # Reasoning models (e.g. gpt-oss-120b) may return content=None with reasoning in a
    # separate field when max_tokens is too low to finish the thinking phase.
    content: str = msg.get("content") or msg.get("reasoning") or ""
    if not content:
        raise _RetryableHTTPError("Empty content and reasoning in response — likely max_tokens too low")
    prompt_tokens: int = data.get("usage", {}).get("prompt_tokens", 0)
    return content, prompt_tokens


def _call_with_fallback(model: str, messages: list[dict[str, str]], **kwargs: Any) -> tuple[str, int]:
    """Try model; on 429 walk the fallback chain for that provider."""
    try:
        return _call_llm_raw(model, messages, **kwargs)
    except _RateLimitError:
        prefix = _provider_prefix(model)
        fallbacks = _FALLBACK_CHAIN.get(prefix, [])
        for fb_model in fallbacks:
            try:
                return _call_llm_raw(fb_model, messages, **kwargs)
            except _RateLimitError:
                continue
        raise RuntimeError(
            f"All providers rate-limited for model '{model}' and its fallbacks {fallbacks}."
        )


def call_llm(model: str, messages: list[dict[str, str]], **kwargs: Any) -> str:
    content, _ = _call_with_fallback(model, messages, **kwargs)
    return content


def call_llm_with_usage(
    model: str, messages: list[dict[str, str]], **kwargs: Any
) -> tuple[str, int]:
    return _call_with_fallback(model, messages, **kwargs)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def call_llm_json(
    model: str,
    messages: list[dict[str, str]],
    max_parse_retries: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    working_messages = list(messages)
    last_err: Exception | None = None

    for _ in range(max_parse_retries):
        raw = call_llm(model=model, messages=working_messages, **kwargs)
        cleaned = _strip_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_err = exc
            working_messages = working_messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your response was not valid JSON. Error: {exc}. "
                        "Reply with valid JSON only — no markdown fences, no prose."
                    ),
                },
            ]

    raise ValueError(
        f"Failed to get valid JSON after {max_parse_retries} attempts. "
        f"Last error: {last_err}"
    )
