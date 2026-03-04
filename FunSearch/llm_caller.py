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
"""
import truststore
truststore.inject_into_ssl()

import os
import asyncio
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
    "google": "gemma-3-27b-it", # "gemini-2.5-flash", #
}

MAX_OUTPUT_TOKENS = 5_000


class LLMCaller:
    """
    Thin async wrapper around Anthropic and Google Gemini APIs.
    Supports multi-image prompts for visual feedback on model fits.

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
        """
        Send a prompt with optional images and return the response text.

        Images are appended after the prompt text in the order given.
        To interleave images with text segments, use call_interleaved().

        Parameters
        ----------
        prompt      : text prompt to send
        image_paths : optional list of PNG image paths to include

        Returns
        -------
        Response text, or None on failure.
        """
        images = _load_images(image_paths)
        if self.provider == "anthropic":
            return await self._call_anthropic([(prompt, images)])
        else:
            return await self._call_google([(prompt, images)])

    async def call_interleaved(
        self,
        segments: list[tuple[str, list[Path] | None]],
    ) -> str | None:
        """
        Send a prompt built from interleaved text and image segments.

        Each segment is a (text, image_paths) tuple. This allows images to
        appear between text sections rather than all at the end — useful for
        associating each evaluation figure with its corresponding program.

        Parameters
        ----------
        segments : list of (text, image_paths) tuples. image_paths may be None.

        Returns
        -------
        Response text, or None on failure.
        """
        loaded = [(text, _load_images(paths)) for text, paths in segments]
        if self.provider == "anthropic":
            return await self._call_anthropic(loaded)
        else:
            return await self._call_google(loaded)

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
    # Provider implementations
    # ------------------------------------------------------------------

    async def _call_anthropic(
        self,
        segments: list[tuple[str, list[bytes]]],
    ) -> str | None:
        """
        Build an Anthropic content list from interleaved text/image segments.
        Images are inserted before their associated text segment.
        """
        try:
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
        except Exception as e:
            print(f"[llm_caller] Anthropic error: {e}")
            return None

    async def _call_google(
        self,
        segments: list[tuple[str, list[bytes]]],
    ) -> str | None:
        """
        Build a Google contents list from interleaved text/image segments.
        Images are inserted before their associated text segment.
        """
        try:
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


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else "google"
    print(f"Testing provider : {provider}")
    print(f"Model            : {DEFAULT_MODELS[provider]}\n")

    caller = LLMCaller(provider=provider, temperature=0.5)
    response = caller.call_sync("Love")

    if response:
        print(f"Response: {response}")
    else:
        print("No response — check your API key and network connection.")