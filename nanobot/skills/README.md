# nanobot Skills

This directory contains built-in skills that extend nanobot's capabilities.

## Skill Format

Each skill is a directory containing a `SKILL.md` file with:
- YAML frontmatter (name, description, metadata)
- Markdown instructions for the agent

## Attribution

These skills are adapted from [OpenClaw](https://github.com/openclaw/openclaw)'s skill system.
The skill format and metadata structure follow OpenClaw's conventions to maintain compatibility.

## Available Skills

| Skill | Description |
|-------|-------------|
| `1password` | Set up and use 1Password CLI (op). Use when installing the CLI, enabling desktop app integration, signing in (single or multi-account), or reading/injecting/running secrets via op. |
| `bird` | X/Twitter CLI for reading, searching, posting, and engagement via cookies. |
| `blogwatcher` | Monitor blogs and RSS/Atom feeds for updates using the blogwatcher CLI. |
| `blucli` | BluOS CLI (blu) for discovery, playback, grouping, and volume. |
| `bluebubbles` | Use when you need to send or manage iMessages via BlueBubbles (recommended iMessage integration). Calls go through the generic message tool with channel="bluebubbles". |
| `camsnap` | Capture frames or clips from RTSP/ONVIF cameras. |
| `canvas` |  |
| `clawhub` | Use the ClawHub CLI to search, install, update, and publish agent skills from clawhub.com. Use when you need to fetch new skills on the fly, sync installed skills to latest or a specific version, or publish new/updated skill folders with the npm-installed clawhub CLI. |
| `coding-agent` | Run Codex CLI, Claude Code, OpenCode, or Pi Coding Agent via background process for programmatic control. |
| `discord` | Use when you need to control Discord from OpenClaw via the discord tool: send messages, react, post or upload stickers, upload emojis, run polls, manage threads/pins/search, create/edit/delete channels and categories, fetch permissions or member/role/channel info, set bot presence/activity, or handle moderation actions in Discord DMs or channels. |
| `eightctl` | Control Eight Sleep pods (status, temperature, alarms, schedules). |
| `food-order` | Reorder Foodora orders + track ETA/status with ordercli. Never confirm without explicit user approval. Triggers: order food, reorder, track ETA. |
| `gemini` | Gemini CLI for one-shot Q&A, summaries, and generation. |
| `gifgrep` | Search GIF providers with CLI/TUI, download results, and extract stills/sheets. |
| `github` | Interact with GitHub using the `gh` CLI. Use `gh issue`, `gh pr`, `gh run`, and `gh api` for issues, PRs, CI runs, and advanced queries. |
| `gog` | Google Workspace CLI for Gmail, Calendar, Drive, Contacts, Sheets, and Docs. |
| `goplaces` | Query Google Places API (New) via the goplaces CLI for text search, place details, resolve, and reviews. Use for human-friendly place lookup or JSON output for scripts. |
| `healthcheck` | Host security hardening and risk-tolerance configuration for OpenClaw deployments. Use when a user asks for security audits, firewall/SSH/update hardening, risk posture, exposure review, OpenClaw cron scheduling for periodic checks, or version status checks on a machine running OpenClaw (laptop, workstation, Pi, VPS). |
| `himalaya` | CLI to manage emails via IMAP/SMTP. Use `himalaya` to list, read, write, reply, forward, search, and organize emails from the terminal. Supports multiple accounts and message composition with MML (MIME Meta Language). |
| `local-places` | Search for places (restaurants, cafes, etc.) via Google Places API proxy on localhost. |
| `mcporter` | Use the mcporter CLI to list, configure, auth, and call MCP servers/tools directly (HTTP or stdio), including ad-hoc servers, config edits, and CLI/type generation. |
| `nano-banana-pro` | Generate or edit images via Gemini 3 Pro Image (Nano Banana Pro). |
| `nano-pdf` | Edit PDFs with natural-language instructions using the nano-pdf CLI. |
| `notion` | Notion API for creating and managing pages, databases, and blocks. |
| `obsidian` | Work with Obsidian vaults (plain Markdown notes) and automate via obsidian-cli. |
| `openai-image-gen` | Batch-generate images via OpenAI Images API. Random prompt sampler + `index.html` gallery. |
| `openai-whisper` | Local speech-to-text with the Whisper CLI (no API key). |
| `openai-whisper-api` | Transcribe audio via OpenAI Audio Transcriptions API (Whisper). |
| `openhue` | Control Philips Hue lights/scenes via the OpenHue CLI. |
| `oracle` | Best practices for using the oracle CLI (prompt + file bundling, engines, sessions, and file attachment patterns). |
| `ordercli` | Foodora-only CLI for checking past orders and active order status (Deliveroo WIP). |
| `sag` | ElevenLabs text-to-speech with mac-style say UX. |
| `session-logs` | Search and analyze your own session logs (older/parent conversations) using jq. |
| `sherpa-onnx-tts` | Local text-to-speech via sherpa-onnx (offline, no cloud) |
| `skill-creator` | Create or update AgentSkills. Use when designing, structuring, or packaging skills with scripts, references, and assets. |
| `slack` | Use when you need to control Slack from OpenClaw via the slack tool, including reacting to messages or pinning/unpinning items in Slack channels or DMs. |
| `songsee` | Generate spectrograms and feature-panel visualizations from audio with the songsee CLI. |
| `sonoscli` | Control Sonos speakers (discover/status/play/volume/group). |
| `spotify-player` | Terminal Spotify playback/search via spogo (preferred) or spotify_player. |
| `summarize` | Summarize or extract text/transcripts from URLs, podcasts, and local files (great fallback for “transcribe this YouTube/video”). |
| `tmux` | Remote-control tmux sessions for interactive CLIs by sending keystrokes and scraping pane output. |
| `trello` | Manage Trello boards, lists, and cards via the Trello REST API. |
| `video-frames` | Extract frames or short clips from videos using ffmpeg. |
| `voice-call` | Start voice calls via the OpenClaw voice-call plugin. |
| `wacli` | Send WhatsApp messages to other people or search/sync WhatsApp history via the wacli CLI (not for normal user chats). |
| `weather` | Get current weather and forecasts (no API key required). |
