"""LLM client for RHOAI 3 llm-d inference gateway.

Models are served via LLMInferenceService through the OpenShift Gateway API.
Supports three connection modes:

  1. Direct URL override (most flexible):
     Set LLM_VISION_URL, LLM_STATE_URL, LLM_PUZZLE_URL, LLM_ADVISOR_URL
     to full endpoint URLs.  Ignores gateway/namespace/model name entirely.
     Example: LLM_STATE_URL=http://qwen-7b.my-ns.svc:8000/v1/chat/completions

  2. Direct workload service (default for in-cluster):
     LLM_USE_DIRECT=true builds URLs from model name + namespace:
     {protocol}://{model}-{service_suffix}.{namespace}.{cluster_domain}:{port}/v1/chat/completions
     All components are configurable via env vars.

  3. Gateway mode:
     LLM_USE_DIRECT=false routes through the llm-d inference gateway:
     {gateway_url}/{namespace}/{model}/v1/chat/completions

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

# Direct service URL components — configurable for different clusters/namespaces
DIRECT_SERVICE_SUFFIX = os.getenv("LLM_SERVICE_SUFFIX", "kserve-workload-svc")
DIRECT_PORT = os.getenv("LLM_SERVICE_PORT", "8000")
DIRECT_PROTOCOL = os.getenv("LLM_SERVICE_PROTOCOL", "https")
DIRECT_CLUSTER_DOMAIN = os.getenv("LLM_CLUSTER_DOMAIN", "svc.cluster.local")

# Model name → LLMInferenceService name mapping
MODEL_ROUTES = {
    "vision": os.getenv("LLM_VISION_MODEL", "qwen25-vl-32b"),
    "dialog": os.getenv("LLM_DIALOG_MODEL", "qwen25-7b"),
    "puzzle": os.getenv("LLM_PUZZLE_MODEL", "qwen25-32b"),
    "state": os.getenv("LLM_STATE_MODEL", "qwen25-7b"),
    "advisor": os.getenv("LLM_ADVISOR_MODEL", "qwen25-32b"),
}

# Per-route URL overrides — if set, bypass gateway/direct pattern entirely.
# This is the most flexible option for cross-namespace or cross-cluster models.
ROUTE_URL_OVERRIDES = {
    "vision": os.getenv("LLM_VISION_URL", ""),
    "dialog": os.getenv("LLM_DIALOG_URL", ""),
    "puzzle": os.getenv("LLM_PUZZLE_URL", ""),
    "state": os.getenv("LLM_STATE_URL", ""),
    "advisor": os.getenv("LLM_ADVISOR_URL", ""),
}

# Fallback routes — if the primary model is unavailable after all retries,
# try these alternative routes.  Used when qwen25-32b is scaled to 0.
ROUTE_FALLBACKS: dict[str, str] = {
    "puzzle": os.getenv("LLM_PUZZLE_FALLBACK", "state"),    # 32b → 7b
    "advisor": os.getenv("LLM_ADVISOR_FALLBACK", "state"),  # 32b → 7b
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
        # Disable SSL verification for direct mode (self-signed certs) or
        # when LLM_VERIFY_SSL is explicitly set to false
        verify_ssl = os.getenv("LLM_VERIFY_SSL", "").lower()
        if verify_ssl:
            ssl_verify = verify_ssl in ("true", "1", "yes")
        else:
            ssl_verify = not use_direct
        self._client = httpx.Client(timeout=self._timeout, verify=ssl_verify)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Public route methods
    # ------------------------------------------------------------------

    def vision(
        self,
        image_b64: str | list[str],
        game_state: dict[str, Any],
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Call /vision route with frame screenshot(s) + state.

        Accepts a single base64 image or a list for multi-frame analysis.
        When multiple images are provided, [Frame N/M] markers are
        interleaved so the model can reference specific frames.

        Returns parsed JSON matching SCHEMAS.md vision output.
        """
        if prompt is None:
            prompt = (
                "Analyze this game frame. Output JSON with: HUD state, dialog flags, "
                "interactables, hazards, Link position, room_id."
            )

        images = [image_b64] if isinstance(image_b64, str) else image_b64

        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(game_state)},
        ]
        for idx, img in enumerate(images):
            if len(images) > 1:
                content.append(
                    {"type": "text", "text": f"[Frame {idx + 1}/{len(images)}]"}
                )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                }
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
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

    def advise(self, prompt: str) -> dict[str, Any]:
        """Call /advisor route for reward parameter tuning advice.

        Returns parsed JSON with multipliers and rationale.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a reward engineering advisor for a reinforcement learning "
                    "agent. Analyze the training stats and output JSON with reward "
                    "parameter multipliers. Output ONLY valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._call("advisor", messages, max_tokens=512, temperature=0.3)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(
        self,
        route: str,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.1,
        _is_fallback: bool = False,
    ) -> dict[str, Any]:
        """Make an HTTP call with retries and JSON parsing.

        URL resolution order:
          1. Per-route URL override (LLM_{ROUTE}_URL env var) — most flexible
          2. Direct workload service (LLM_USE_DIRECT=true) — in-cluster default
          3. Gateway mode (LLM_USE_DIRECT=false) — through llm-d gateway

        If all retries fail and a ROUTE_FALLBACKS entry exists, the call
        is retried on the fallback route (e.g. puzzle → state when
        qwen25-32b is scaled to 0).
        """
        model_name = MODEL_ROUTES.get(route.lstrip("/"), route.lstrip("/"))
        url = self._resolve_url(route, model_name)
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

        # Try fallback route if primary route exhausted retries
        clean_route = route.lstrip("/")
        if not _is_fallback and clean_route in ROUTE_FALLBACKS:
            fallback = ROUTE_FALLBACKS[clean_route]
            logger.warning(
                "LLM %s failed — trying fallback route '%s' (%s)",
                route, fallback, MODEL_ROUTES.get(fallback, fallback),
            )
            return self._call(
                fallback, messages,
                max_tokens=max_tokens, temperature=temperature,
                _is_fallback=True,
            )

        logger.error("LLM %s failed after %d retries: %s", route, self._max_retries, last_error)
        return {"error": str(last_error), "route": route}

    def _resolve_url(self, route: str, model_name: str) -> str:
        """Resolve the endpoint URL for a given route.

        Priority:
          1. Per-route URL override (LLM_{ROUTE}_URL) — full URL, no pattern
          2. Direct workload service — build from model + namespace + service components
          3. Gateway — route through the llm-d inference gateway
        """
        # Check for per-route URL override first
        override = ROUTE_URL_OVERRIDES.get(route.lstrip("/"), "")
        if override:
            return override

        if self._use_direct:
            return (
                f"{DIRECT_PROTOCOL}://{model_name}-{DIRECT_SERVICE_SUFFIX}"
                f".{self._namespace}.{DIRECT_CLUSTER_DOMAIN}:{DIRECT_PORT}"
                f"/v1/chat/completions"
            )
        return f"{self._gateway}/{self._namespace}/{model_name}/v1/chat/completions"

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
