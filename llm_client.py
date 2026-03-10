"""
LLM Client – Provider-specific clients for structured LLM calls.

Supports:
  - OpenAI  (gpt-4o, etc.)     — via openai SDK
  - Gemini  (gemini-2.5-flash) — via google-genai SDK (native, hits Google servers)

Designed to be stateless: each call is independent, no conversation history.
Both providers use JSON Schema structured output so the LLM is forced to
return valid JSON matching our Pydantic schema.

Usage:
    client = create_llm_client(provider="openai")  # or "gemini"
    result = await client.structured_completion(
        system_prompt="...",
        user_prompt="...",
        response_schema=MyPydanticModel,
    )
"""
from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from typing import Any, Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


# ─── OpenAI Strict Schema Compatibility ───────────────────────────────────────


def _make_openai_strict_schema(schema: dict) -> dict:
    """
    Transform a Pydantic JSON Schema into one compatible with OpenAI's
    strict structured output mode. OpenAI requires:
      1. 'additionalProperties': false on every object
      2. All properties listed in 'required' (use anyOf with null for optionals)

    This recursively processes the schema and all $defs.
    """
    schema = dict(schema)

    if "$defs" in schema:
        schema["$defs"] = {
            name: _fix_object_schema(defn)
            for name, defn in schema["$defs"].items()
        }

    schema = _fix_object_schema(schema)
    return schema


def _fix_object_schema(schema: dict) -> dict:
    """Fix a single object schema for OpenAI strict mode."""
    schema = dict(schema)

    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())

        new_props = {}
        for prop_name, prop_schema in schema["properties"].items():
            new_props[prop_name] = _fix_property_schema(prop_schema)
        schema["properties"] = new_props

    return schema


def _fix_property_schema(prop: dict) -> dict:
    """Fix a single property schema, handling anyOf/optional patterns."""
    prop = dict(prop)

    if "$ref" in prop:
        return {"$ref": prop["$ref"]}

    if "anyOf" in prop:
        new_any = []
        for variant in prop["anyOf"]:
            if isinstance(variant, dict):
                variant = _fix_property_schema(variant)
            new_any.append(variant)
        return {"anyOf": new_any}

    if prop.get("type") == "object" and "properties" in prop:
        return _fix_object_schema(prop)

    if prop.get("type") == "array" and "items" in prop:
        prop["items"] = _fix_property_schema(prop["items"])
        return prop

    return prop


# ─── Abstract Base ─────────────────────────────────────────────────────────────


class BaseLLMClient(ABC):
    """Abstract base for LLM clients."""

    provider: str
    model: str

    @abstractmethod
    async def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        ...


# ─── OpenAI Client ─────────────────────────────────────────────────────────────


class OpenAIClient(BaseLLMClient):
    """OpenAI-native client using the openai SDK."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.provider = "openai"
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

        if not self.api_key:
            raise ValueError("No OPENAI_API_KEY found. Set it in your .env file.")

        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=self.api_key)

    async def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        schema = response_schema.model_json_schema()
        schema = _make_openai_strict_schema(schema)

        print(f"🤖 Calling openai/{self.model}...")

        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "autofill_response",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty response")

        parsed = json.loads(content)
        print(f"   ✅ Got structured response ({len(content):,} chars)")
        return parsed


# ─── Gemini Client ─────────────────────────────────────────────────────────────


class GeminiClient(BaseLLMClient):
    """
    Google Gemini-native client using the google-genai SDK.
    Hits Google's servers directly (ai.google.dev), NOT OpenAI.
    Uses your Google AI Studio API key.
    """

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        self.provider = "gemini"
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

        if not self.api_key:
            raise ValueError("No GEMINI_API_KEY found. Set it in your .env file.")

        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    async def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        from google.genai import types

        print(f"🤖 Calling gemini/{self.model}...")

        # Gemini's native SDK supports Pydantic models directly for structured output
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=user_prompt)],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        content = response.text
        if not content:
            raise ValueError("Gemini returned empty response")

        parsed = json.loads(content)
        print(f"   ✅ Got structured response ({len(content):,} chars)")
        return parsed


# ─── Factory ───────────────────────────────────────────────────────────────────


def create_llm_client(
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
) -> BaseLLMClient:
    """
    Create an LLM client for the given provider.

    Args:
        provider: "openai" or "gemini"
        model:    Model name. Defaults to gpt-4o (OpenAI) or gemini-2.5-flash (Gemini).
        api_key:  Optional API key override.
    """
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o", api_key=api_key)
    elif provider == "gemini":
        return GeminiClient(model=model or "gemini-2.5-flash", api_key=api_key)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Supported: openai, gemini")
