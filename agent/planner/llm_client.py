"""LLM client for RHOAI 3 llm-d inference gateway.

Models are served via LLMInferenceService through the OpenShift Gateway API.
Internal URL pattern:
  http://<gateway-svc>/<namespace>/<model-name>/v1/chat/completions

Model mapping:
  vision — qwen25-vl-32b (multimodal frame analysis)
  dialog — qwen25-7b     (dialog navigation)
  puzzle — qwen25-32b    (puzzle solving)
  state  — qwen25-7b     (state analysis, shares with dialog)

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
    "LLM_GATEWAY_URL",
    "http://openshift-ai-inference-openshift-default.openshift-ingress.svc.cluster.local",
)
DEFAULT_NAMESPACE = os.getenv("LLM_NAMESPACE", "zelda-rl")

# Use direct workload service URLs for in-cluster calls (bypasses EPP,
# avoids body-forwarding issues, lower latency for RL training).
# Set LLM_USE_DIRECT=true to enable (default for in-cluster).
USE_DIRECT = os.getenv("LLM_USE_DIRECT", "true").lower() in ("true", "1", "yes")

# Model name → LLMInferenceService name mapping
MODEL_ROUTES = {
    "vision": os.getenv("LLM_VISION_MODEL", "qwen25-vl-32b"),
    "dialog": os.getenv("LLM_DIALOG_MODEL", "qwen25-7b"),
    "puzzle": os.getenv("LLM_PUZZLE_MODEL", "qwen25-32b"),
    "state": os.getenv("LLM_STATE_MODEL", "qwen25-7b"),
}


class LLMClient:
    """HTTP client for RHOAI 3 llm-d inference gateway."""

    def __init__(
        self,
        gateway_url: str = DEFAULT_GATEWAY,
        namespace: str = DEFAULT_NAMESPACE,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        seed: int = 42,
        use_direct: bool = USE_DIRECT,
    ):
        self._gateway = gateway_url.rstrip("/")
        self._namespace = namespace
        self._max_retries = max_retries
        self._timeout = timeout_s
        self._seed = seed
        self._use_direct = use_direct
        # Direct mode uses HTTPS with self-signed certs (vLLM secure serving)
        self._client = httpx.Client(timeout=self._timeout, verify=not use_direct)

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
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]
        return self._call("vision", messages, max_tokens=512, temperature=0.1)

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
        return self._call("dialog", messages, max_tokens=128, temperature=0.1)

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
        return self._call("puzzle", messages, max_tokens=256, temperature=0.2)

    def state(self, game_state: dict[str, Any]) -> dict[str, Any]:
        """Call /state route for state analysis."""
        messages = [
            {
                "role": "system",
                "content": "Analyze game state and output structured JSON assessment.",
            },
            {"role": "user", "content": json.dumps(game_state)},
        ]
        return self._call("state", messages, max_tokens=256, temperature=0.1)

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
        Direct mode: https://{model}-kserve-workload-svc.{ns}.svc:8000/v1/chat/completions
        Gateway mode: http://{gateway}/{namespace}/{model}/v1/chat/completions
        """
        model_name = MODEL_ROUTES.get(route.lstrip("/"), route.lstrip("/"))
        if self._use_direct:
            url = (
                f"https://{model_name}-kserve-workload-svc"
                f".{self._namespace}.svc.cluster.local:8000"
                f"/v1/chat/completions"
            )
        else:
            url = f"{self._gateway}/{self._namespace}/{model_name}/v1/chat/completions"
        session_id = self._session_id(messages)

        payload = {
            "model": model_name,
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
                if resp.status_code != 200:
                    logger.warning(
                        "LLM %s returned %d: %s",
                        route, resp.status_code, resp.text[:500],
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
