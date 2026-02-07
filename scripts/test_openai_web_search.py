#!/usr/bin/env python3
"""
Smoke test for OpenAI Responses API web_search support.

Usage:
  OPENAI_API_KEY=... OPENAI_API_BASE=... python3 scripts/test_openai_web_search.py \
    --model gpt-5.2 --query "What is a positive news story today?" --include-sources
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import ssl


def _env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return ""


def _build_responses_url(api_base: str) -> str:
    base = (api_base or "https://api.openai.com").rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _extract_output_text(resp: dict) -> str:
    parts: list[str] = []
    for item in resp.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _extract_sources(resp: dict) -> list[dict]:
    sources: list[dict] = []
    for item in resp.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        for src in action.get("sources") or []:
            if isinstance(src, dict):
                sources.append(src)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI web_search smoke test")
    parser.add_argument(
        "--model",
        default=_env("OPENAI_MODEL", "NANOBOT_MODEL") or "gpt-5.2",
        help="Model name (default: gpt-5.2)",
    )
    parser.add_argument(
        "--query",
        default="What is a positive news story from today?",
        help="Search query prompt",
    )
    parser.add_argument(
        "--search-context-size",
        default="medium",
        choices=["low", "medium", "high"],
        help="web_search search_context_size",
    )
    parser.add_argument(
        "--allowed-domains",
        default="",
        help="Comma-separated allowed domains (optional)",
    )
    parser.add_argument(
        "--include-sources",
        action="store_true",
        help="Include web_search sources in the response",
    )
    parser.add_argument(
        "--external-web-access",
        default=None,
        help="true/false to set external_web_access",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Max output tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--ca-bundle",
        default="",
        help="Path to a CA bundle to trust (optional)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (not recommended)",
    )

    args = parser.parse_args()

    api_key = _env(
        "OPENAI_API_KEY",
        "NANOBOT_PROVIDERS__OPENAI__API_KEY",
    )
    api_base = _env(
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "NANOBOT_PROVIDERS__OPENAI__API_BASE",
    )

    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY.", file=sys.stderr)
        return 2

    try:
        external_web_access = _parse_bool(args.external_web_access)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    tools = [{
        "type": "web_search",
        "search_context_size": args.search_context_size,
    }]

    if args.allowed_domains:
        allowed = [d.strip() for d in args.allowed_domains.split(",") if d.strip()]
        if allowed:
            tools[0]["filters"] = {"allowed_domains": allowed}

    if external_web_access is not None:
        tools[0]["external_web_access"] = external_web_access

    payload = {
        "model": args.model,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": args.query}],
        }],
        "tools": tools,
        "tool_choice": "auto",
        "max_output_tokens": args.max_output_tokens,
    }
    payload["temperature"] = args.temperature

    if args.include_sources:
        payload["include"] = ["web_search_call.action.sources"]

    url = _build_responses_url(api_base)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    print(f"POST {url}")
    print(f"model={args.model} query={args.query!r}")

    try:
        if args.insecure:
            context = ssl._create_unverified_context()
        elif args.ca_bundle:
            context = ssl.create_default_context(cafile=args.ca_bundle)
        else:
            try:
                import certifi  # type: ignore
                context = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                context = ssl.create_default_context()

        with urllib.request.urlopen(req, timeout=args.timeout, context=context) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as err:
        error_body = err.read().decode("utf-8", errors="replace")
        print(f"HTTP {err.code} {err.reason}", file=sys.stderr)
        print(error_body, file=sys.stderr)
        return 1
    except urllib.error.URLError as err:
        print(f"Network error: {err}", file=sys.stderr)
        return 1
    except json.JSONDecodeError:
        print("Failed to parse JSON response", file=sys.stderr)
        print(raw, file=sys.stderr)
        return 1

    text = _extract_output_text(data)
    sources = _extract_sources(data)

    if text:
        print("\n--- Output ---")
        print(text)
    else:
        print("\nNo output_text found. Raw response keys:", list(data.keys()))

    if sources:
        print("\n--- Sources ---")
        for src in sources:
            title = src.get("title") or "(no title)"
            url = src.get("url") or ""
            print(f"- {title} {url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
