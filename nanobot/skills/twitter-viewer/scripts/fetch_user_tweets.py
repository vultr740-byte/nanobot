#!/usr/bin/env python3
"""Fetch user tweets from twitter-viewer.com with optional pagination."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://www.twitter-viewer.com/api/x/user-tweets"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
REFERER = "https://www.twitter-viewer.com/"


def _build_url(username: str, cursor: str) -> str:
    clean = username.strip()
    if clean.startswith("@"):
        clean = clean[1:]
    params = {"username": clean, "cursor": cursor or ""}
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener()


def _fetch(url: str, timeout: float) -> dict[str, Any]:
    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": USER_AGENT,
        "referer": REFERER,
    }
    req = urllib.request.Request(url, headers=headers)
    opener = _build_opener()
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code}: {detail}".strip()) from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    text = body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Response was not valid JSON") from exc


def _write_output(data: dict[str, Any], pretty: bool, out_path: str | None) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)
    if out_path:
        Path(out_path).write_text(payload, encoding="utf-8")
    else:
        print(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch user tweets from twitter-viewer.com")
    parser.add_argument("--username", required=True, help="Twitter/X username (without @)")
    parser.add_argument("--cursor", default="", help="Pagination cursor (empty for first page)")
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout (seconds)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--out", default=None, help="Write JSON to a file instead of stdout")
    args = parser.parse_args()

    url = _build_url(args.username, args.cursor)
    try:
        data = _fetch(url, args.timeout)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _write_output(data, args.pretty, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
