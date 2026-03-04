"""
llm_caller.py
-------------
Async LLM API client supporting Anthropic and Google Gemini.

Usage:
    caller = LLMCaller(provider="google")
    response = await caller.call("your prompt", image_paths=[Path("fig1.png"), Path("fig2.png")])

    # Synchronous (for testing only):
    response = caller.call_sync("your prompt", image_paths=[...])

Multiple images are supported and interleaved with prompt text segments
when using build_prompt() from prompt_builder.py.

API keys are loaded from a .env file:
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...

Rate limiting
-------------
429 RESOURCE_EXHAUSTED responses are retried automatically. The retry delay
is parsed directly from the error message (e.g. "retry in 23s") and a small
buffer is added. Falls back to exponential backoff if no delay is parseable.
Max retries is controlled by MAX_RETRIES (default 6).
"""

import asyncio
import base64
import logging
import os
import re
import sys
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.ERROR)

Provider = Literal["anthropic", "google"]

DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemma-3-27b-it",
}

MAX_OUTPUT_TOKENS = 5_000
MAX_RETRIES = 6          # max attempts before giving up
RETRY_BUFFER_SECS = 2.0  # extra seconds added on top of the server-suggested delay


class LLMCaller:
    """
    Thin async wrapper around Anthropic and Google Gemini APIs.
    Supports multi-image prompts for visual feedback on model fits.

    429 rate-limit errors are retried automatically, waiting the server-
    suggested delay (plus RETRY_BUFFER_SECS) between attempts.

    Parameters
    ----------
    provider    : "anthropic" or "google" (default: "google")
    model       : model name; defaults to a sensible cheap model per provider
    temperature : sampling temperature
    """

    def __init__(
        self,
        provider: Provider = "google",
        model: str | None = None,
        temperature: float = 1.0,
    ):
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.temperature = temperature
        self._client = self._init_client()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def call(
        self,
        prompt: str,
        image_paths: list[Path] | None = None,
    ) -> str | None:
        images = _load_images(image_paths)
        if self.provider == "anthropic":
            return await self._call_with_retry(self._call_anthropic, [(prompt, images)])
        else:
            return await self._call_with_retry(self._call_google, [(prompt, images)])

    async def call_interleaved(
        self,
        segments: list[tuple[str, list[Path] | None]],
    ) -> str | None:
        loaded = [(text, _load_images(paths)) for text, paths in segments]
        if self.provider == "anthropic":
            return await self._call_with_retry(self._call_anthropic, loaded)
        else:
            return await self._call_with_retry(self._call_google, loaded)

    def call_sync(
        self,
        prompt: str,
        image_paths: list[Path] | None = None,
    ) -> str | None:
        """Synchronous wrapper — for testing only, not for use in async loops."""
        return asyncio.run(self.call(prompt, image_paths=image_paths))

    def call_interleaved_sync(
        self,
        segments: list[tuple[str, list[Path] | None]],
    ) -> str | None:
        """Synchronous wrapper around call_interleaved."""
        return asyncio.run(self.call_interleaved(segments))

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    async def _call_with_retry(self, fn, *args, **kwargs) -> str | None:
        """
        Call fn(*args, **kwargs) and retry on 429 / quota errors.

        Waits the server-suggested delay (parsed from the error message)
        plus RETRY_BUFFER_SECS. Falls back to exponential backoff
        (10s, 20s, 40s …) if no delay hint is available.
        """
        backoff = 10.0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if not _is_rate_limit_error(e):
                    # Non-rate-limit error — log and give up immediately
                    print(f"[llm_caller] Non-retryable error: {e}")
                    return None

                suggested = _parse_retry_delay(str(e))
                wait = (suggested + RETRY_BUFFER_SECS) if suggested else backoff
                print(
                    f"[llm_caller] 429 rate limit (attempt {attempt}/{MAX_RETRIES}). "
                    f"Waiting {wait:.1f}s ..."
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, 120.0)  # cap at 2 minutes

        print(f"[llm_caller] Giving up after {MAX_RETRIES} retries.")
        return None

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _call_anthropic(
        self,
        segments: list[tuple[str, list[bytes]]],
    ) -> str | None:
        """
        Build an Anthropic content list from interleaved text/image segments.
        Raises on error so _call_with_retry can catch and classify it.
        """
        content: list[dict] = []
        for text, images in segments:
            for img_bytes in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(img_bytes).decode(),
                    },
                })
            content.append({"type": "text", "text": text})

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    async def _call_google(
        self,
        segments: list[tuple[str, list[bytes]]],
    ) -> str | None:
        """
        Build a Google contents list from interleaved text/image segments.
        Raises on error so _call_with_retry can catch and classify it.
        """
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        contents: list = []
        for text, images in segments:
            for img_bytes in images:
                contents.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                )
            contents.append(text)

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _init_client(self):
        if self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not found in environment or .env file")
            return anthropic.AsyncAnthropic(api_key=api_key)

        elif self.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY not found in environment or .env file")
            return genai.Client(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {self.provider!r}.")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_images(image_paths: list[Path] | None) -> list[bytes]:
    """Load a list of image paths into bytes. Skips missing files with a warning."""
    if not image_paths:
        return []
    result = []
    for p in image_paths:
        p = Path(p)
        if not p.exists():
            print(f"[llm_caller] Image not found, skipping: {p}")
            continue
        result.append(p.read_bytes())
    return result


def _parse_retry_delay(error_str: str) -> float | None:
    """
    Extract the suggested retry delay from a 429 error message.

    Handles formats like:
      "Please retry in 23.558356064s"
      "retryDelay: '23s'"
      "retry_delay { seconds: 23 }"
    Returns the delay in seconds, or None if unparseable.
    """
    # "retry in 23.5s" or "retry in 23s"
    m = re.search(r'retry[^\d]*(\d+(?:\.\d+)?)\s*s', error_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # plain number of seconds anywhere
    m = re.search(r'(\d+(?:\.\d+)?)\s*seconds?', error_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _is_rate_limit_error(e: Exception) -> bool:
    """Return True if the exception looks like a 429 / quota error."""
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "google"
    print(f"Testing provider : {provider}")
    print(f"Model            : {DEFAULT_MODELS[provider]}\n")

    caller = LLMCaller(provider=provider, temperature=0.5)
    response = caller.call_sync("Say hello in one sentence.")

    if response:
        print(f"Response: {response}")
    else:
        print("No response — check your API key and network connection.")