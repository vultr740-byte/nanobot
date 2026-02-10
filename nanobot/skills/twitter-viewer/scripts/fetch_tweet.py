#!/usr/bin/env python3
"""Fetch a single tweet by tweetId from twitter-viewer.com."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://www.twitter-viewer.com/api/x/tweet"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
REFERER = "https://www.twitter-viewer.com/"

TWEET_ID_PATTERNS = [
    re.compile(r"tweetId=([0-9]{5,})", re.IGNORECASE),
    re.compile(r"/status/([0-9]{5,})", re.IGNORECASE),
    re.compile(r"/statuses/([0-9]{5,})", re.IGNORECASE),
]


def _build_url(tweet_id: str) -> str:
    params = {"tweetId": tweet_id}
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


def _extract_tweet_id(text: str) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.isdigit():
        return stripped
    for pattern in TWEET_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    numbers = re.findall(r"\b\d{15,19}\b", text)
    if numbers:
        return numbers[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a single tweet from twitter-viewer.com")
    parser.add_argument("--tweet-id", dest="tweet_id", help="Tweet ID to fetch")
    parser.add_argument(
        "--message",
        help="Message text containing a tweet URL or tweetId to parse",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="Request timeout (seconds)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--out", default=None, help="Write JSON to a file instead of stdout")
    args = parser.parse_args()

    tweet_id = args.tweet_id
    if not tweet_id and args.message:
        tweet_id = _extract_tweet_id(args.message)

    if not tweet_id:
        print("Error: tweetId not found. Provide --tweet-id or --message containing a tweet URL.", file=sys.stderr)
        return 2

    url = _build_url(tweet_id)
    try:
        data = _fetch(url, args.timeout)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _write_output(data, args.pretty, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
