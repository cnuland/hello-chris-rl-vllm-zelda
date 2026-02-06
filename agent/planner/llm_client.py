"""LLM client for llm-d inference gateway routes.

Routes:
  /vision  — Qwen2.5-VL-32B (multimodal frame analysis)
  /dialog  — Qwen2.5-7B (dialog navigation)
  /puzzle  — Qwen2.5-32B (puzzle solving)
  /state   — Qwen2.5-7B (state analysis, shares with dialog)

Each route uses:
  - Retries with exponential backoff
  - Deterministic decoding (low temperature)
  - JSON grammar enforcement (structured output)
  - Session affinity via x-session-id header (KV-cache reuse)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY = os.getenv(
    "LLM_GATEWAY_URL", "http://llm-d-gateway.zelda-rl.svc.cluster.local:8080"
)


class LLMClient:
    """HTTP client for llm-d gateway routes."""

    def __init__(
        self,
        gateway_url: str = DEFAULT_GATEWAY,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        seed: int = 42,
    ):
        self._gateway = gateway_url.rstrip("/")
        self._max_retries = max_retries
        self._timeout = timeout_s
        self._seed = seed
        self._client = httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Public route methods
    # ------------------------------------------------------------------

    def vision(
        self,
        image_b64: str,
        game_state: dict[str, Any],
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Call /vision route with frame screenshot + state.

        Returns parsed JSON matching SCHEMAS.md vision output.
        """
        if prompt is None:
            prompt = (
                "Analyze this game frame. Output JSON with: HUD state, dialog flags, "
                "interactables, hazards, Link position, room_id."
            )

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(game_state)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ]
        return self._call("/vision", messages, max_tokens=512, temperature=0.1)

    def dialog(self, game_state: dict[str, Any], prompt: str | None = None) -> dict[str, Any]:
        """Call /dialog route for dialog navigation.

        Returns parsed JSON matching SCHEMAS.md Dialog output.
        """
        if prompt is None:
            prompt = (
                "You are a dialog operator for Zelda: Oracle of Seasons. "
                "Advance text, choose options, confirm YES. "
                "Output JSON: {intent, presses, guard, confidence}."
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(game_state)},
        ]
        return self._call("/dialog", messages, max_tokens=128, temperature=0.1)

    def puzzle(self, game_state: dict[str, Any], prompt: str | None = None) -> dict[str, Any]:
        """Call /puzzle route for puzzle solving.

        Returns parsed JSON matching SCHEMAS.md Puzzle output.
        """
        if prompt is None:
            prompt = (
                "You are a puzzle operator for Zelda: Oracle of Seasons. "
                "Decide subgoal and macro sequence. "
                "Output JSON: {subgoal, macros, fallback, confidence}."
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(game_state)},
        ]
        return self._call("/puzzle", messages, max_tokens=256, temperature=0.2)

    def state(self, game_state: dict[str, Any]) -> dict[str, Any]:
        """Call /state route for state analysis."""
        messages = [
            {
                "role": "system",
                "content": "Analyze game state and output structured JSON assessment.",
            },
            {"role": "user", "content": json.dumps(game_state)},
        ]
        return self._call("/state", messages, max_tokens=256, temperature=0.1)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(
        self,
        route: str,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Make an HTTP call with retries and JSON parsing.

        Uses OpenAI-compatible /v1/chat/completions format.
        """
        url = f"{self._gateway}{route}/v1/chat/completions"
        session_id = self._session_id(messages)

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": self._seed,
            "response_format": {"type": "json_object"},
        }

        last_error = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-session-id": session_id,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_json(content)
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                last_error = e
                wait = 2**attempt
                logger.warning(
                    "LLM %s attempt %d failed: %s (retry in %ds)",
                    route,
                    attempt + 1,
                    e,
                    wait,
                )
                time.sleep(wait)

        logger.error("LLM %s failed after %d retries: %s", route, self._max_retries, last_error)
        return {"error": str(last_error), "route": route}

    def _parse_json(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM output, with repair for common issues."""
        content = content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try extracting first JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse LLM JSON: %s", content[:200])
            return {"raw": content, "parse_error": True}

    def _session_id(self, messages: list[dict]) -> str:
        """Generate session ID from message content for KV-cache affinity."""
        content = ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                content += msg["content"][:64]
        prefix = content[:64].encode()
        return f"cache-{hashlib.md5(prefix).hexdigest()[:16]}"
