"""CLI commands for nanobot."""

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from loguru import logger

from nanobot import __version__, __logo__

app = typer.Typer(
    name="nanobot",
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
_FILE_LOGGING_READY = False


def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """nanobot - Personal AI Assistant."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard():
    """Initialize nanobot configuration and workspace."""
    from nanobot.config.loader import get_config_path, save_config
    from nanobot.config.schema import Config
    from nanobot.utils.helpers import get_workspace_path
    
    config_path = get_config_path()
    
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit()
    
    # Create default config
    config = Config()
    save_config(config)
    console.print(f"[green]✓[/green] Created config at {config_path}")
    
    # Create workspace
    workspace = get_workspace_path()
    console.print(f"[green]✓[/green] Created workspace at {workspace}")
    
    # Create default bootstrap files
    _create_workspace_templates(workspace)
    
    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    console.print(f"  1. Add your API key to [cyan]{config_path}[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print("  2. Chat: [cyan]nanobot agent -m \"Hello!\"[/cyan]")
    console.print("\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]")




def _load_workspace_template(filename: str, fallback: str) -> str:
    """Load a workspace template from repo if available; otherwise use fallback."""
    repo_root = Path(__file__).parent.parent.parent
    template_path = repo_root / "workspace" / filename
    if template_path.exists():
        try:
            return template_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return fallback


def _setup_file_logging() -> None:
    global _FILE_LOGGING_READY
    if _FILE_LOGGING_READY:
        return
    from nanobot.utils.helpers import get_data_path
    log_dir = get_data_path() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dir / "nanobot.log"), rotation="00:00", retention="1 day", level="INFO")
    _FILE_LOGGING_READY = True


def _create_workspace_templates(workspace: Path):
    """Create default workspace template files (only if missing)."""
    workspace.mkdir(parents=True, exist_ok=True)
    templates = {
        "AGENTS.md": _load_workspace_template(
            "AGENTS.md",
            """# Agent Instructions

You are a helpful AI assistant. Be concise, accurate, and friendly.

## Guidelines

- Always explain what you're doing before taking actions
- Ask for clarification when the request is ambiguous
- Use tools to help accomplish tasks
- Remember important information in your memory files

## Tools Available

You have access to:
- File operations (read, write, edit, list)
- Shell commands (exec)
- Web access (search, fetch)
- Messaging (message)
- Background tasks (spawn)

## Subagent Policy

- For any task likely to take >30 seconds, require external IO (network/files), or needs multiple steps, spawn a subagent.
- Each subagent must handle exactly one task; do not chain unrelated tasks in a single subagent.
- Keep the main agent responsive; if a subagent runs long, send a brief status update every 20-40 seconds.
- If a subagent has no response after ~2 minutes, notify the user and offer a fallback plan or retry.

## Memory

- Use `memory/` directory for daily notes
- Use `MEMORY.md` for long-term information

## Scheduled Reminders

When the user asks for a reminder at a specific time, schedule a cron job.

- Use `--message` for a simple reminder text (agent_turn).
- Use `--exec` for running scripts/commands directly (exec).

Examples:
```
nanobot cron add --name "reminder" --message "Your message" --at "YYYY-MM-DDTHH:MM:SS" --deliver --to "USER_ID" --channel "CHANNEL"
nanobot cron add --name "job" --exec "python /path/task.py" --cron "0 9 * * *" --deliver --to "USER_ID" --channel "CHANNEL"
```
Get USER_ID and CHANNEL from the current session (e.g., `8281248569` and `telegram` from `telegram:8281248569`).

Do NOT just write reminders to MEMORY.md - that won't trigger actual notifications.

## Heartbeat Tasks

HEARTBEAT.md is checked every 30 minutes. You can manage periodic tasks by editing this file:

- Add a task: Use edit_file to append new tasks to HEARTBEAT.md
- Remove a task: Use edit_file to remove completed or obsolete tasks
- Rewrite tasks: Use write_file to completely rewrite the task list

Task format examples:
```
- [ ] Check calendar and remind of upcoming events
- [ ] Scan inbox for urgent emails
- [ ] Check weather forecast for today
```

When the user asks you to add a recurring/periodic task, update HEARTBEAT.md instead of creating a one-time reminder. Keep the file small to minimize token usage.
""",
        ),
        "SOUL.md": _load_workspace_template(
            "SOUL.md",
"""# Soul

I am a helpful teammate.

## Personality

- Helpful and friendly
- Concise and to the point
- Curious and eager to learn

## Values

- Accuracy over speed
- User privacy and safety
- Transparency in actions
""",
        ),
        "USER.md": _load_workspace_template(
            "USER.md",
            """# User

Information about the user goes here.

## Preferences

- Communication style: (casual/formal)
- Timezone: (your timezone)
- Language: (your preferred language)
""",
        ),
        "TOOLS.md": _load_workspace_template(
            "TOOLS.md",
            """# Available Tools

This document describes the tools available to the agent.

## File Operations

### read_file
Read the contents of a file.
```
read_file(path: str) -> str
```

### write_file
Write content to a file (creates parent directories if needed).
```
write_file(path: str, content: str) -> str
```

### edit_file
Edit a file by replacing specific text.
```
edit_file(path: str, old_text: str, new_text: str) -> str
```

### list_dir
List contents of a directory.
```
list_dir(path: str) -> str
```

## Shell Execution

### exec
Execute a shell command and return output.
```
exec(command: str, working_dir: str = None) -> str
```
By default, commands run via bash -lc when bash is available; otherwise they fall back to /bin/sh.

Safety notes:
- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- Optional restrictToWorkspace config to limit paths

## Web Access

### web_search
Search the web (Brave Search API when configured).
```
web_search(query: str, count: int = 5) -> str
```

### web_fetch
Fetch and extract main content from a URL.
```
web_fetch(url: str, extractMode: str = "markdown", maxChars: int = 50000) -> str
```

## Communication

### message
Send a message to the user (used internally).
```
message(content: str, media: list[str] = None, channel: str = None, chat_id: str = None) -> str
```

## Background Tasks

### spawn
Spawn a subagent to handle a task in the background.
```
spawn(task: str, label: str = None) -> str
```

## Scheduled Reminders (Cron)

Use nanobot cron add to schedule either agent messages or shell commands.

### Agent reminder (message)
```bash
# Every day at 9am
nanobot cron add --name "morning" --message "Good morning!" --cron "0 9 * * *"

# Every 2 hours
nanobot cron add --name "water" --message "Drink water!" --every 7200
```

### One-time reminder
```bash
# At a specific time (ISO format)
nanobot cron add --name "meeting" --message "Meeting starts now!" --at "2025-01-31T15:00:00"
```

### Execute a command
```bash
# Run a script every minute
nanobot cron add --name "time" --exec "python /path/cron_time_notify.py" --cron "* * * * *"

# Run with a working directory
nanobot cron add --name "report" --exec "python report.py" --cwd "/path/to/project" --cron "0 9 * * *"
```

### Manage reminders
```bash
nanobot cron list              # List all jobs
nanobot cron remove <job_id>   # Remove a job
```

## Heartbeat Task Management

The HEARTBEAT.md file in the workspace is checked every 30 minutes.
Use file operations to manage periodic tasks.
""",
        ),
        "HEARTBEAT.md": _load_workspace_template(
            "HEARTBEAT.md",
            """# Heartbeat Tasks

This file is checked every 30 minutes by your agent.
Add tasks below that you want the agent to work on periodically.

If this file has no tasks (only headers and comments), the agent will skip the heartbeat.

## Active Tasks

<!-- Add your periodic tasks below this line -->


## Completed

<!-- Move completed tasks here or delete them -->
""",
        ),
    }
    
    for filename, content in templates.items():
        file_path = workspace / filename
        if not file_path.exists():
            file_path.write_text(content)
            console.print(f"  [dim]Created {filename}[/dim]")
    
    # Create memory directory and MEMORY.md
    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text("""# Long-term Memory

This file stores important information that should persist across sessions.

## User Information

(Important facts about the user)

## Preferences

(User preferences learned over time)

## Important Notes

(Things to remember)
""")
        console.print("  [dim]Created memory/MEMORY.md[/dim]")


def _build_web_search_settings(config):
    """Build web search provider settings from config."""
    web_cfg = config.tools.web.search
    provider = web_cfg.provider
    active_provider = config.get_active_provider_name()
    provider_explicit = bool(os.getenv("NANOBOT_TOOLS__WEB__SEARCH__PROVIDER"))
    if not provider_explicit:
        try:
            from nanobot.config.loader import get_config_path
            config_path = get_config_path()
            if config_path.exists():
                with open(config_path) as f:
                    raw = json.load(f)
                search_cfg = raw.get("tools", {}).get("web", {}).get("search", {})
                provider_explicit = isinstance(search_cfg, dict) and "provider" in search_cfg
        except Exception:
            provider_explicit = False

    # Auto-default to OpenAI web search when only OpenAI is viable.
    if (
        provider == "brave"
        and active_provider == "openai"
        and config.providers.openai.api_key
        and not provider_explicit
    ):
        provider = "openai"

    openai_settings = None
    if provider == "openai":
        openai_settings = {
            "search_context_size": web_cfg.search_context_size,
            "allowed_domains": web_cfg.allowed_domains,
            "include_sources": web_cfg.include_sources,
        }
        if web_cfg.external_web_access is not None:
            openai_settings["external_web_access"] = web_cfg.external_web_access
        if web_cfg.user_location:
            openai_settings["user_location"] = web_cfg.user_location.model_dump(exclude_none=True)

    return provider, openai_settings


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the nanobot gateway."""
    from nanobot.config.loader import load_config, get_data_dir, get_config_path
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.agent.loop import AgentLoop
    from nanobot.channels.manager import ChannelManager
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.heartbeat.service import HeartbeatService
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    _setup_file_logging()
    console.print(f"{__logo__} Starting nanobot gateway on port {port}...")
    
    config = load_config()
    config_path = get_config_path()
    if config_path.exists():
        console.print(f"[yellow]Config file detected:[/yellow] {config_path}")
    _create_workspace_templates(config.workspace_path)
    
    # Create components
    bus = MessageBus()
    
    # Create provider (supports OpenRouter, Anthropic, OpenAI, Bedrock)
    api_key = config.get_api_key()
    api_base = config.get_api_base()
    model = config.agents.defaults.model
    is_bedrock = model.startswith("bedrock/")
    
    console.print(f"[dim]Model:[/dim] {model}")
    console.print(f"[dim]API base:[/dim] {api_base or 'default'}")
    console.print(f"[dim]API key set:[/dim] {'yes' if api_key else 'no'}")

    if not api_key and not is_bedrock:
        console.print("[red]Error: No API key configured.[/red]")
        console.print(f"Set one in {config_path} under providers.openrouter.apiKey")
        raise typer.Exit(1)
    
    provider_config = config.get_active_provider_config()
    force_chat_completions = bool(provider_config and provider_config.force_chat_completions)
    strip_temperature = bool(provider_config and provider_config.strip_temperature)
    web_search_provider, openai_web_search_config = _build_web_search_settings(config)
    provider_name = config.get_active_provider_name()
    provider = LiteLLMProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=config.agents.defaults.model,
        force_chat_completions=force_chat_completions,
        strip_temperature=strip_temperature,
        provider_name=provider_name,
        openai_web_search=web_search_provider == "openai",
        openai_web_search_config=openai_web_search_config,
    )
    
    # Create agent
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        brave_api_key=config.tools.web.search.api_key or None,
        web_search_provider=web_search_provider,
        exec_config=config.tools.exec,
    )
    
    # Create cron service
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        response: str | None
        if job.payload.kind == "exec":
            command = job.payload.command or job.payload.message
            if not command:
                response = "Error: cron exec job missing command"
            else:
                params = {"command": command}
                if job.payload.working_dir:
                    params["working_dir"] = job.payload.working_dir
                response = await agent.tools.execute("exec", params)
        else:
            response = await agent.process_direct(
                job.payload.message,
                session_key=f"cron:{job.id}"
            )
        # Optionally deliver to channel
        if job.payload.deliver and job.payload.to:
            from nanobot.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "whatsapp",
                chat_id=job.payload.to,
                content=response or ""
            ))
        return response
    
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path, on_job=on_cron_job)
    
    # Create heartbeat service
    async def on_heartbeat(prompt: str) -> str:
        """Execute heartbeat through the agent."""
        return await agent.process_direct(prompt, session_key="heartbeat")
    
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        on_heartbeat=on_heartbeat,
        interval_s=30 * 60,  # 30 minutes
        enabled=True
    )
    
    # Create channel manager
    channels = ChannelManager(config, bus)
    
    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")
    
    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")
    
    console.print(f"[green]✓[/green] Heartbeat: every 30m")
    
    async def run():
        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()
    
    asyncio.run(run())




# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:default", "--session", "-s", help="Session ID"),
):
    """Interact with the agent directly."""
    from nanobot.config.loader import load_config
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.agent.loop import AgentLoop
    
    _setup_file_logging()
    config = load_config()
    _create_workspace_templates(config.workspace_path)
    
    api_key = config.get_api_key()
    api_base = config.get_api_base()
    model = config.agents.defaults.model
    is_bedrock = model.startswith("bedrock/")

    if not api_key and not is_bedrock:
        console.print("[red]Error: No API key configured.[/red]")
        raise typer.Exit(1)

    bus = MessageBus()
    provider_config = config.get_active_provider_config()
    force_chat_completions = bool(provider_config and provider_config.force_chat_completions)
    strip_temperature = bool(provider_config and provider_config.strip_temperature)
    web_search_provider, openai_web_search_config = _build_web_search_settings(config)
    provider_name = config.get_active_provider_name()
    provider = LiteLLMProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=config.agents.defaults.model,
        force_chat_completions=force_chat_completions,
        strip_temperature=strip_temperature,
        provider_name=provider_name,
        openai_web_search=web_search_provider == "openai",
        openai_web_search_config=openai_web_search_config,
    )
    
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        brave_api_key=config.tools.web.search.api_key or None,
        web_search_provider=web_search_provider,
        exec_config=config.tools.exec,
    )
    
    if message:
        # Single message mode
        async def run_once():
            response = await agent_loop.process_direct(message, session_id)
            console.print(f"\n{__logo__} {response}")
        
        asyncio.run(run_once())
    else:
        # Interactive mode
        console.print(f"{__logo__} Interactive mode (Ctrl+C to exit)\n")
        
        async def run_interactive():
            while True:
                try:
                    user_input = console.input("[bold blue]You:[/bold blue] ")
                    if not user_input.strip():
                        continue
                    
                    response = await agent_loop.process_direct(user_input, session_id)
                    console.print(f"\n{__logo__} {response}\n")
                except KeyboardInterrupt:
                    console.print("\nGoodbye!")
                    break
        
        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from nanobot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("Configuration", style="yellow")

    # WhatsApp
    wa = config.channels.whatsapp
    table.add_row(
        "WhatsApp",
        "✓" if wa.enabled else "✗",
        wa.bridge_url
    )

    # Telegram
    tg = config.channels.telegram
    tg_config = f"token: {tg.token[:10]}..." if tg.token else "[dim]not configured[/dim]"
    table.add_row(
        "Telegram",
        "✓" if tg.enabled else "✗",
        tg_config
    )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess
    
    # User's bridge location
    from nanobot.utils.helpers import get_data_path
    user_bridge = get_data_path() / "bridge"
    
    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge
    
    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)
    
    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # nanobot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)
    
    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge
    
    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall nanobot")
        raise typer.Exit(1)
    
    console.print(f"{__logo__} Setting up bridge...")
    
    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))
    
    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("  Building...")
        subprocess.run(["npm", "run", "build"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)
    
    return user_bridge


@channels_app.command("login")
def channels_login():
    """Link device via QR code."""
    import subprocess
    
    bridge_dir = _get_bridge_dir()
    
    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")
    
    try:
        subprocess.run(["npm", "start"], cwd=bridge_dir, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]npm not found. Please install Node.js.[/red]")


# ============================================================================
# Cron Commands
# ============================================================================

cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all: bool = typer.Option(False, "--all", "-a", help="Include disabled jobs"),
):
    """List scheduled jobs."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    jobs = service.list_jobs(include_disabled=all)
    
    if not jobs:
        console.print("No scheduled jobs.")
        return
    
    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Schedule")
    table.add_column("Status")
    table.add_column("Next Run")
    
    import time
    for job in jobs:
        # Format schedule
        if job.schedule.kind == "every":
            sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
        elif job.schedule.kind == "cron":
            sched = job.schedule.expr or ""
        else:
            sched = "one-time"
        
        # Format next run
        next_run = ""
        if job.state.next_run_at_ms:
            next_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(job.state.next_run_at_ms / 1000))
            next_run = next_time
        
        status = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"
        
        table.add_row(job.id, job.name, job.payload.kind, sched, status, next_run)
    
    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    message: str | None = typer.Option(None, "--message", "-m", help="Message for agent"),
    exec_command: str | None = typer.Option(None, "--exec", help="Shell command to execute"),
    cwd: str | None = typer.Option(None, "--cwd", help="Working directory for --exec"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    cron_expr: str = typer.Option(None, "--cron", "-c", help="Cron expression (e.g. '0 9 * * *')"),
    at: str = typer.Option(None, "--at", help="Run once at time (ISO format)"),
    deliver: bool = typer.Option(False, "--deliver", "-d", help="Deliver response to channel"),
    to: str = typer.Option(None, "--to", help="Recipient for delivery"),
    channel: str = typer.Option(None, "--channel", help="Channel for delivery (e.g. 'telegram', 'whatsapp')"),
):
    """Add a scheduled job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronSchedule
    
    # Determine schedule type
    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Error: Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)
    
    if exec_command and message:
        console.print("[red]Error: Use --message or --exec, not both[/red]")
        raise typer.Exit(1)
    if not exec_command and not message:
        console.print("[red]Error: Must specify --message or --exec[/red]")
        raise typer.Exit(1)

    payload_kind = "exec" if exec_command else "agent_turn"
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.add_job(
        name=name,
        schedule=schedule,
        message=message or "",
        payload_kind=payload_kind,
        command=exec_command or "",
        working_dir=cwd,
        deliver=deliver,
        to=to,
        channel=channel,
    )
    
    console.print(f"[green]✓[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
):
    """Remove a scheduled job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    if service.remove_job(job_id):
        console.print(f"[green]✓[/green] Removed job {job_id}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID"),
    disable: bool = typer.Option(False, "--disable", help="Disable instead of enable"),
):
    """Enable or disable a job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.enable_job(job_id, enabled=not disable)
    if job:
        status = "disabled" if disable else "enabled"
        console.print(f"[green]✓[/green] Job '{job.name}' {status}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    force: bool = typer.Option(False, "--force", "-f", help="Run even if disabled"),
):
    """Manually run a job."""
    from nanobot.config.loader import get_data_dir, load_config
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.agent.tools.shell import ExecTool
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.agent.loop import AgentLoop
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    config = load_config()
    _create_workspace_templates(config.workspace_path)

    # Lightweight exec tool for exec-type cron jobs.
    exec_tool = ExecTool(
        working_dir=str(config.workspace_path),
        timeout=config.tools.exec.timeout,
        restrict_to_workspace=config.tools.exec.restrict_to_workspace,
    )

    agent_loop: AgentLoop | None = None

    async def on_cron_job(job: CronJob) -> str | None:
        nonlocal agent_loop, last_response
        if job.payload.kind == "exec":
            command = job.payload.command or job.payload.message
            if not command:
                last_response = "Error: cron exec job missing command"
                return last_response
            last_response = await exec_tool.execute(command, working_dir=job.payload.working_dir)
            return last_response

        # agent_turn path
        if agent_loop is None:
            api_key = config.get_api_key()
            api_base = config.get_api_base()
            model = config.agents.defaults.model
            is_bedrock = model.startswith("bedrock/")
            if not api_key and not is_bedrock:
                last_response = "Error: No API key configured."
                return last_response

            bus = MessageBus()
            provider_config = config.get_active_provider_config()
            force_chat_completions = bool(provider_config and provider_config.force_chat_completions)
            strip_temperature = bool(provider_config and provider_config.strip_temperature)
            web_search_provider, openai_web_search_config = _build_web_search_settings(config)
            provider_name = config.get_active_provider_name()
            provider = LiteLLMProvider(
                api_key=api_key,
                api_base=api_base,
                default_model=config.agents.defaults.model,
                force_chat_completions=force_chat_completions,
                strip_temperature=strip_temperature,
                provider_name=provider_name,
                openai_web_search=web_search_provider == "openai",
                openai_web_search_config=openai_web_search_config,
            )

            agent_loop = AgentLoop(
                bus=bus,
                provider=provider,
                workspace=config.workspace_path,
                brave_api_key=config.tools.web.search.api_key or None,
                web_search_provider=web_search_provider,
                exec_config=config.tools.exec,
            )

        last_response = await agent_loop.process_direct(
            job.payload.message, session_key=f"cron:{job.id}"
        )
        return last_response

    service = CronService(store_path, on_job=on_cron_job)
    last_response: str | None = None

    async def run():
        result = await service.run_job(job_id, force=force)
        return result

    if asyncio.run(run()):
        console.print(f"[green]✓[/green] Job executed")
        if last_response:
            console.print(last_response)
    else:
        console.print(f"[red]Failed to run job {job_id}[/red]")


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status():
    """Show nanobot status."""
    from nanobot.config.loader import load_config, get_config_path

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        console.print(f"Model: {config.agents.defaults.model}")
        
        # Check API keys
        has_openrouter = bool(config.providers.openrouter.api_key)
        has_anthropic = bool(config.providers.anthropic.api_key)
        has_openai = bool(config.providers.openai.api_key)
        has_gemini = bool(config.providers.gemini.api_key)
        has_vllm = bool(config.providers.vllm.api_base)
        
        console.print(f"OpenRouter API: {'[green]✓[/green]' if has_openrouter else '[dim]not set[/dim]'}")
        console.print(f"Anthropic API: {'[green]✓[/green]' if has_anthropic else '[dim]not set[/dim]'}")
        console.print(f"OpenAI API: {'[green]✓[/green]' if has_openai else '[dim]not set[/dim]'}")
        console.print(f"Gemini API: {'[green]✓[/green]' if has_gemini else '[dim]not set[/dim]'}")
        vllm_status = f"[green]✓ {config.providers.vllm.api_base}[/green]" if has_vllm else "[dim]not set[/dim]"
        console.print(f"vLLM/Local: {vllm_status}")


if __name__ == "__main__":
    app()
