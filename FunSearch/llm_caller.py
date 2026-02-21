"""
llm_caller.py
-------------
Async LLM API client supporting Anthropic and Google Gemini.

Usage:
    from FunSearch.llm_caller import LLMCaller

    caller = LLMCaller(provider="google")
    response = await caller.call("your prompt here")

    # With an image:
    response = await caller.call("describe this", image_path=Path("figure.png"))

API keys are loaded from a .env file in the project root:
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...

Synchronous calls are available via caller.call_sync(...) for testing.
"""

import os
import asyncio
import base64
import logging
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
    "google": "gemma-3-27b-it", #"gemini-2.5-flash",
}

MAX_OUTPUT_TOKENS = 5_000


class LLMCaller:
    """
    Thin async wrapper around Anthropic and Google Gemini APIs.

    Parameters
    ----------
    provider    : "anthropic" or "google" (default: "google" — free tier available)
    model       : model name string; defaults to a sensible cheap model per provider
    temperature : sampling temperature (higher = more creative / exploratory)
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
        image_path: Path | None = None,
    ) -> str | None:
        """
        Send a prompt asynchronously and return the response text.

        Parameters
        ----------
        prompt     : text prompt to send
        image_path : optional path to a PNG image to include alongside the prompt

        Returns
        -------
        Response text string, or None on failure.
        """
        image_bytes = _load_image(image_path)
        if self.provider == "anthropic":
            return await self._call_anthropic(prompt, image_bytes)
        else:
            return await self._call_google(prompt, image_bytes)

    def call_sync(self, prompt: str, image_path: Path | None = None) -> str | None:
        """
        Synchronous wrapper around call() — convenient for testing.
        Avoid using this inside an already-running event loop.
        """
        return asyncio.run(self.call(prompt, image_path=image_path))

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _call_anthropic(self, prompt: str, image_bytes: bytes | None) -> str | None:
        try:
            content: list = [{"type": "text", "text": prompt}]
            if image_bytes is not None:
                content.insert(0, {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image_bytes).decode(),
                    },
                })
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=self.temperature,
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"[llm_caller] Anthropic error: {e}")
            return None

    async def _call_google(self, prompt: str, image_bytes: bytes | None) -> str | None:
        try:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            contents: list = [prompt]
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
            raise ValueError(f"Unknown provider: {self.provider!r}. Choose 'anthropic' or 'google'.")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_image(image_path: Path | None) -> bytes | None:
    """Read image bytes from a path, or return None if no path given."""
    if image_path is None:
        return None
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[llm_caller] Image not found: {image_path}")
        return None
    return image_path.read_bytes()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else "google"
    print(f"Testing provider : {provider}")
    print(f"Model            : {DEFAULT_MODELS[provider]}\n")

    caller = LLMCaller(provider=provider, temperature=0.5)
    response = caller.call_sync("Sending love.")

    if response:
        print(f"Response: {response}")
    else:
        print("No response — check your API key and network connection.")