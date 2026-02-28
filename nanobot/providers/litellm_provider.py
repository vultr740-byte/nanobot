"""LiteLLM provider implementation for multi-provider support."""

import json
import json_repair
import os
import secrets
import string
from typing import Any

import httpx
import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway


# Standard OpenAI chat-completion message keys plus reasoning_content for
# thinking-enabled models (Kimi k2.5, DeepSeek-R1, etc.).
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"})
_ALNUM = string.ascii_letters + string.digits

def _short_tool_id() -> str:
    """Generate a 9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
        openai_web_search: bool = False,
        openai_web_search_config: dict[str, Any] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self.provider_name = provider_name
        self.openai_web_search = openai_web_search
        self.openai_web_search_config = openai_web_search_config or {}
        
        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True
    
    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        if not spec.env_key:
            # OAuth/provider-only specs (for example: openai_codex)
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            model = self._canonicalize_explicit_prefix(model, spec.name, spec.litellm_prefix)
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"

        return model

    @staticmethod
    def _canonicalize_explicit_prefix(model: str, spec_name: str, canonical_prefix: str) -> str:
        """Normalize explicit provider prefixes like `github-copilot/...`."""
        if "/" not in model:
            return model
        prefix, remainder = model.split("/", 1)
        if prefix.lower().replace("-", "_") != spec_name:
            return model
        return f"{canonical_prefix}/{remainder}"
    
    def _supports_cache_control(self, model: str) -> bool:
        """Return True when the provider supports cache_control on content blocks."""
        if self._gateway is not None:
            return self._gateway.supports_prompt_caching
        spec = find_by_model(model)
        return spec is not None and spec.supports_prompt_caching

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Return copies of messages and tools with cache_control injected."""
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if isinstance(content, str):
                    new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                else:
                    new_content = list(content)
                    new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
                new_messages.append({**msg, "content": new_content})
            else:
                new_messages.append(msg)

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}

        return new_messages, new_tools

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
    
    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-standard keys and ensure assistant messages have a content key."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
            # Strict providers require "content" even when assistant only has tool_calls
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)

        if self._supports_cache_control(original_model):
            messages, tools = self._apply_cache_control(messages, tools)

        # Clamp max_tokens to at least 1 — negative or zero values cause
        # LiteLLM to reject the request with "max_tokens must be at least 1".
        max_tokens = max(1, max_tokens)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_key directly — more reliable than env vars alone
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        if self.openai_web_search and self._should_use_openai_responses(model):
            payload = self._build_responses_payload(
                messages=self._sanitize_messages(self._sanitize_empty_content(messages)),
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            try:
                response = await self._responses_httpx(payload)
                return self._parse_responses_dict(response)
            except Exception as e:
                return LLMResponse(
                    content=f"Error calling LLM: {str(e)}",
                    finish_reason="error",
                )
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json_repair.loads(args)
                
                tool_calls.append(ToolCallRequest(
                    id=_short_tool_id(),
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        reasoning_content = getattr(message, "reasoning_content", None) or None
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )

    @staticmethod
    def _build_responses_url(api_base: str) -> str:
        base = api_base.rstrip("/") if api_base else "https://api.openai.com"
        if base.endswith("/responses"):
            return base
        if base.endswith("/v1"):
            return f"{base}/responses"
        return f"{base}/v1/responses"

    def _should_use_openai_responses(self, model: str) -> bool:
        if self.provider_name is not None:
            return self.provider_name == "openai"
        if self._gateway is not None:
            return False
        model_lower = model.lower()
        return "gpt" in model_lower or model_lower.startswith("o")

    def _build_responses_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        instructions, input_items = self._messages_to_responses_input(messages)

        payload: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if instructions:
            payload["instructions"] = instructions

        tools_payload: list[dict[str, Any]] = []
        if self.openai_web_search:
            tools_payload.append(self._build_openai_web_search_tool())
        if tools:
            tools_payload.extend(self._convert_tools_for_responses(tools))
        if tools_payload:
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"

        include_sources = bool(self.openai_web_search_config.get("include_sources"))
        if include_sources:
            payload["include"] = ["web_search_call.action.sources"]

        return payload

    def _build_openai_web_search_tool(self) -> dict[str, Any]:
        tool_type = self.openai_web_search_config.get("tool_type") or "web_search"
        tool: dict[str, Any] = {"type": tool_type}

        search_context_size = self.openai_web_search_config.get("search_context_size")
        if search_context_size:
            tool["search_context_size"] = search_context_size

        allowed_domains = self.openai_web_search_config.get("allowed_domains") or []
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains}

        user_location = self.openai_web_search_config.get("user_location")
        if user_location:
            tool["user_location"] = user_location

        if "external_web_access" in self.openai_web_search_config:
            tool["external_web_access"] = bool(self.openai_web_search_config["external_web_access"])

        return tool

    def _convert_tools_for_responses(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                fn = tool.get("function")
                if isinstance(fn, dict):
                    name = fn.get("name")
                    if not name:
                        continue
                    out: dict[str, Any] = {"type": "function", "name": name}
                    description = fn.get("description")
                    parameters = fn.get("parameters")
                    if description:
                        out["description"] = description
                    if parameters is not None:
                        out["parameters"] = parameters
                    converted.append(out)
                    continue
                if tool.get("name"):
                    converted.append(tool)
                continue
            converted.append(tool)
        return converted

    async def _responses_httpx(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._build_responses_url(self.api_base or "")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            headers.update(self.extra_headers)

        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    def _messages_to_responses_input(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instructions = None
        items: list[dict[str, Any]] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")

            if idx == 0 and role == "system":
                instructions = self._coerce_text(content)
                continue

            if role == "tool":
                call_id = msg.get("tool_call_id") or ""
                output = msg.get("content") or ""
                if call_id:
                    items.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    })
                continue

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", "")
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=True)
                    item: dict[str, Any] = {
                        "type": "function_call",
                        "name": fn.get("name") or "",
                        "arguments": args,
                    }
                    call_id = tc.get("id")
                    if call_id:
                        item["call_id"] = call_id
                    items.append(item)

                if content:
                    items.append(self._message_input_item(role, content))
                continue

            items.append(self._message_input_item(role, content))

        return instructions, items

    def _message_input_item(self, role: str | None, content: Any) -> dict[str, Any]:
        safe_role = role if role in {"system", "developer", "user", "assistant"} else "user"
        return {
            "type": "message",
            "role": safe_role,
            "content": self._content_to_input_parts(content, safe_role),
        }

    def _content_to_input_parts(self, content: Any, role: str) -> list[dict[str, Any]]:
        text_type = "output_text" if role == "assistant" else "input_text"
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": text_type, "text": content}]
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"}:
                    parts.append({"type": text_type, "text": item.get("text", "")})
                elif item_type == "image_url" and role != "assistant":
                    image = item.get("image_url") or {}
                    url = image.get("url") if isinstance(image, dict) else image
                    if url:
                        parts.append({"type": "input_image", "image_url": url})
                elif item_type in {"input_text", "input_image"} and role != "assistant":
                    parts.append(item)
            if parts:
                return parts
        return [{"type": text_type, "text": self._coerce_text(content)}]

    @staticmethod
    def _coerce_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"text", "input_text", "output_text"}:
                    text = item.get("text")
                    if text:
                        parts.append(text)
            if parts:
                return " ".join(parts)
        return str(content)

    def _parse_responses_dict(self, response: dict[str, Any]) -> LLMResponse:
        output = response.get("output") or []
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content") or []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "output_text" and part.get("text"):
                        content_parts.append(part["text"])
            elif item_type == "function_call":
                args = item.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json_repair.loads(args)
                    except Exception:
                        args = {"raw": args}
                tool_calls.append(ToolCallRequest(
                    id=item.get("call_id") or item.get("id") or _short_tool_id(),
                    name=item.get("name", ""),
                    arguments=args or {},
                ))

        usage = {}
        if isinstance(response.get("usage"), dict):
            usage = response["usage"]

        content = "\n".join([part for part in content_parts if part])
        return LLMResponse(
            content=content if content else None,
            tool_calls=tool_calls,
            finish_reason=response.get("status") or "stop",
            usage=usage,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
