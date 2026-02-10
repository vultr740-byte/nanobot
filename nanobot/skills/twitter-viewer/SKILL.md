---
name: twitter-viewer
description: Fetch X/Twitter data via twitter-viewer.com API, including user timelines (with pagination) and single tweets by tweetId. Use when you need recent tweets for a username, to page through results with a cursor, or to retrieve one specific tweet.
---

# X/Twitter Viewer

Fetch user timelines (paged) and single tweets using the `twitter-viewer.com` API.

## Quick start

- Run the fetch script for the first page (empty cursor):
  - `python3 scripts/fetch_user_tweets.py --username cz_binance --pretty`

## Fetch a single tweet

Use the tweet endpoint when you have a specific tweet ID or a tweet URL in the user message.

- Direct tweet ID:
  - `python3 scripts/fetch_tweet.py --tweet-id 1981404850832494666 --pretty`
- Parse from a message or URL (auto-extracts tweetId):
  - `python3 scripts/fetch_tweet.py --message "https://www.twitter-viewer.com/api/x/tweet?tweetId=1981404850832494666" --pretty`
  - `python3 scripts/fetch_tweet.py --message "https://x.com/cz_binance/status/1981404850832494666" --pretty`

## Pagination workflow

1) Read `data.pagination.nextCursor` from the response.
2) Pass it as `--cursor` to fetch the next page:
   - `python3 scripts/fetch_user_tweets.py --username cz_binance --cursor "<nextCursor>" --pretty`
3) Stop when `data.pagination.hasMore` is `false` or `nextCursor` is empty.

## Output shape

The JSON response includes:

- `data.user`: profile metadata
- `data.tweets`: list of tweets (`id`, `text`, `createdAt`, `author`, ...)
- `data.pagination`: `nextCursor`, `prevCursor`, `hasMore`

## Notes

- This is an unofficial, site-specific API; it may change without notice.
