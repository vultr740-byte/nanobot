#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def build_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def post_json(url: str, api_key: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def extract_tool_calls(response: dict) -> list:
    try:
        choice = response["choices"][0]
        message = choice.get("message") or {}
    except Exception:
        return []
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return tool_calls
    function_call = message.get("function_call")
    if function_call:
        return [function_call]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate tool calling support on a Chat Completions API."
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("COINCOIN_API_BASE", "https://fastapi.key.pro/coincoin/v1"),
        help="API base URL (default from COINCOIN_API_BASE).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("COINCOIN_API_KEY", ""),
        help="API key (default from COINCOIN_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("COINCOIN_MODEL", "gpt-5.2-codex"),
        help="Model name (default from COINCOIN_MODEL).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("COINCOIN_TIMEOUT", "30")),
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--force-tool",
        action="store_true",
        help="Force tool_choice to the test tool instead of auto.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print raw JSON response.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: COINCOIN_API_KEY is required.", file=sys.stderr)
        return 1

    url = build_url(args.api_base)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Return current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a tool-calling test."},
            {
                "role": "user",
                "content": "Call the weather tool for Tokyo in Celsius.",
            },
        ],
        "tools": tools,
        "tool_choice": (
            {"type": "function", "function": {"name": "get_current_weather"}}
            if args.force_tool
            else "auto"
        ),
        "temperature": 0.0,
    }

    try:
        response = post_json(url, args.api_key, payload, args.timeout)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = ""
        print(f"HTTP ERROR: {e.code} {e.reason}", file=sys.stderr)
        if err_body:
            print(err_body, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"REQUEST ERROR: {e}", file=sys.stderr)
        return 1

    if args.print_response:
        print(json.dumps(response, ensure_ascii=True, indent=2))

    tool_calls = extract_tool_calls(response)
    if tool_calls:
        print("OK: tool_calls returned.")
        return 0

    print("FAIL: no tool_calls or function_call in response.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
