"""
llm_caller.py
-------------
Async LLM API client supporting Anthropic and Google Gemini.

Usage:
    from FunSearch.llm_caller import LLMCaller

    caller = LLMCaller(provider="anthropic")        # or "google"
    response = await caller.call("your prompt here")

API keys are loaded from a .env file in the project root:
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...

Synchronous calls are also available via caller.call_sync(...) for testing.
"""

import os
import asyncio
import logging
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# Suppress noisy HTTP logging from API clients
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.ERROR)

Provider = Literal["anthropic", "google"]

# --- Default models ---
DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.0-flash",
}

# --- Token limits ---
MAX_OUTPUT_TOKENS = 5_000


class LLMCaller:
    """
    Thin async wrapper around Anthropic and Google Gemini APIs.

    Parameters
    ----------
    provider    : "anthropic" or "google"
    model       : model name string; defaults to a sensible cheap model per provider
    temperature : sampling temperature (higher = more creative / exploratory)
    """

    def __init__(
        self,
        provider: Provider = "anthropic",
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

    async def call(self, prompt: str, image_bytes: bytes | None = None) -> str | None:
        """
        Send a prompt asynchronously and return the response text.

        Parameters
        ----------
        prompt      : text prompt to send
        image_bytes : optional PNG image bytes to include (Google only for now)

        Returns
        -------
        Response text string, or None on failure.
        """
        if self.provider == "anthropic":
            return await self._call_anthropic(prompt)
        else:
            return await self._call_google(prompt, image_bytes)

    def call_sync(self, prompt: str) -> str | None:
        """
        Synchronous wrapper for testing or non-async contexts.
        """
        return asyncio.run(self.call(prompt))

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _call_anthropic(self, prompt: str) -> str | None:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"[llm_caller] Anthropic error: {e}")
            return None

    async def _call_google(self, prompt: str, image_bytes: bytes | None = None) -> str | None:
        from google.genai import types
        try:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            contents = [prompt]
            if image_bytes is not None:
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))

            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as e:
            print(f"[llm_caller] Google error: {e}")
            return None

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _init_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise EnvironmentError("ANTHROPIC_API_KEY not found in environment or .env file")
                return anthropic.AsyncAnthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

        elif self.provider == "google":
            try:
                from google import genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise EnvironmentError("GOOGLE_API_KEY not found in environment or .env file")
                return genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")

        else:
            raise ValueError(f"Unknown provider: {self.provider!r}. Choose 'anthropic' or 'google'.")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else "anthropic"
    print(f"Testing provider: {provider}")
    print(f"Model: {DEFAULT_MODELS[provider]}\n")

    caller = LLMCaller(provider=provider, temperature=0.5)
    response = caller.call_sync("Reply with exactly three words.")

    if response:
        print(f"Response: {response}")
    else:
        print("No response received — check your API key and network connection.")