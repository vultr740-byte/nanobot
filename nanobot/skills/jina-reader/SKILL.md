---
name: jina-reader
description: Read web pages and PDFs into LLM-friendly text using Jina Reader. Use when the user asks to extract, clean, summarize, or fetch readable content from a URL.
homepage: https://github.com/jina-ai/reader
metadata: { "openclaw": { "emoji": "📖", "requires": { "bins": ["curl"] } } }
---

# Jina Reader

Use Jina Reader endpoints to convert URLs into LLM-friendly text.

## When to use

Trigger this skill when the user asks things like:

- "read this link"
- "extract the article content"
- "turn this page into markdown/text"
- "read this PDF URL"

## Core endpoints

- Read URL: `https://r.jina.ai/<target-url>`

## Quick usage

### 1) Read a URL

```bash
curl -fsSL "https://r.jina.ai/https://example.com"
```

### 2) Read a PDF URL

```bash
curl -fsSL "https://r.jina.ai/https://arxiv.org/pdf/1706.03762.pdf"
```

## Agent workflow

1. Validate the input URL (must include `http://` or `https://`).
2. Fetch content from `r.jina.ai/<url>`.
3. If request fails, retry once.
4. Return concise extracted content or summary based on user ask.
5. If content is too long, provide structured summary first, then offer section-by-section follow-up.

## Safety and reliability notes

- Do not fabricate content if fetch fails.
- Report HTTP/network errors clearly.
- This skill intentionally uses only no-key read flows (`r.jina.ai`) for consistency.
- Respect private/internal URLs: do not fetch localhost or private network targets unless user explicitly requests and environment policy allows it.
- For sensitive pages requiring login/cookies, explain that unauthenticated fetch may be incomplete.
