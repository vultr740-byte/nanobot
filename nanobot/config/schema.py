"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class WhatsAppConfig(BaseModel):
    """WhatsApp channel configuration."""
    enabled: bool = False
    bridge_url: str = "ws://localhost:3001"
    allow_from: list[str] = Field(default_factory=list)  # Allowed phone numbers


class TelegramConfig(BaseModel):
    """Telegram channel configuration."""
    enabled: bool = False
    token: str = ""  # Bot token from @BotFather
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs or usernames
    typing_feedback_delay_s: float = 6.0
    typing_window_s: float = 5.0
    typing_feedback_grace_s: float = 1.0
    typing_feedback_emoji: str = ""
    typing_feedback_long_emoji: str = "👨‍💻"


class ChannelsConfig(BaseModel):
    """Configuration for chat channels."""
    whatsapp: WhatsAppConfig = Field(default_factory=WhatsAppConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class AgentDefaults(BaseModel):
    """Default agent configuration."""
    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20


class AgentsConfig(BaseModel):
    """Agent configuration."""
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""
    api_key: str = ""
    api_base: str | None = None
    force_chat_completions: bool = False
    strip_temperature: bool = False


class ProvidersConfig(BaseModel):
    """Configuration for LLM providers."""
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)


class GatewayConfig(BaseModel):
    """Gateway/server configuration."""
    host: str = "0.0.0.0"
    port: int = 18790


class WebSearchUserLocation(BaseModel):
    """Approximate user location hints for OpenAI web search."""
    type: Literal["approximate"] = "approximate"
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class WebSearchConfig(BaseModel):
    """Web search tool configuration."""
    provider: Literal["brave", "openai", "disabled"] = "brave"
    api_key: str = ""  # Brave Search API key
    max_results: int = 5
    # OpenAI web search options
    search_context_size: Literal["low", "medium", "high"] = "medium"
    allowed_domains: list[str] = Field(default_factory=list)
    include_sources: bool = False
    external_web_access: bool | None = None
    user_location: WebSearchUserLocation | None = None


class WebToolsConfig(BaseModel):
    """Web tools configuration."""
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(BaseModel):
    """Shell exec tool configuration."""
    timeout: int = 60
    restrict_to_workspace: bool = False  # If true, block commands accessing paths outside workspace


class ToolsConfig(BaseModel):
    """Tools configuration."""
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)


class Config(BaseSettings):
    """Root configuration for nanobot."""
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    
    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()
    
    def get_api_key(self) -> str | None:
        """Get API key in priority order: OpenRouter > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM."""
        return (
            self.providers.openrouter.api_key or
            self.providers.anthropic.api_key or
            self.providers.openai.api_key or
            self.providers.gemini.api_key or
            self.providers.zhipu.api_key or
            self.providers.groq.api_key or
            self.providers.vllm.api_key or
            None
        )
    
    def get_api_base(self) -> str | None:
        """Get API base URL if using OpenRouter, OpenAI, Zhipu, or vLLM."""
        if self.providers.openrouter.api_key:
            return self.providers.openrouter.api_base or "https://openrouter.ai/api/v1"
        if self.providers.openai.api_key and self.providers.openai.api_base:
            return self.providers.openai.api_base
        if self.providers.zhipu.api_key:
            return self.providers.zhipu.api_base
        if self.providers.vllm.api_base:
            return self.providers.vllm.api_base
        return None

    def get_active_provider_config(self) -> ProviderConfig | None:
        """Get the active provider config based on API key priority."""
        if self.providers.openrouter.api_key:
            return self.providers.openrouter
        if self.providers.anthropic.api_key:
            return self.providers.anthropic
        if self.providers.openai.api_key:
            return self.providers.openai
        if self.providers.gemini.api_key:
            return self.providers.gemini
        if self.providers.zhipu.api_key:
            return self.providers.zhipu
        if self.providers.groq.api_key:
            return self.providers.groq
        if self.providers.vllm.api_key:
            return self.providers.vllm
        return None

    def get_active_provider_name(self) -> str | None:
        """Get the active provider name based on API key priority."""
        if self.providers.openrouter.api_key:
            return "openrouter"
        if self.providers.anthropic.api_key:
            return "anthropic"
        if self.providers.openai.api_key:
            return "openai"
        if self.providers.gemini.api_key:
            return "gemini"
        if self.providers.zhipu.api_key:
            return "zhipu"
        if self.providers.groq.api_key:
            return "groq"
        if self.providers.vllm.api_key:
            return "vllm"
        return None
    
    class Config:
        env_prefix = "NANOBOT_"
        env_nested_delimiter = "__"
