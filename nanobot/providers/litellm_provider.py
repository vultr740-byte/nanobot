"""LiteLLM provider implementation for multi-provider support."""

import json
import os
from typing import Any

import httpx
import uuid
import litellm
from litellm import acompletion
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, and many other providers through
    a unified interface.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        force_chat_completions: bool = False,
        strip_temperature: bool = False,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.force_chat_completions = force_chat_completions
        self.strip_temperature = strip_temperature
        
        # Detect OpenRouter by api_key prefix or explicit api_base
        self.is_openrouter = (
            (api_key and api_key.startswith("sk-or-")) or
            (api_base and "openrouter" in api_base)
        )
        
        # Track if using custom endpoint (vLLM, etc.)
        self.is_vllm = bool(api_base) and not self.is_openrouter
        
        # Configure LiteLLM based on provider
        if api_key:
            if self.is_openrouter:
                # OpenRouter mode - set key
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif self.is_vllm:
                # vLLM/custom endpoint - uses OpenAI-compatible API
                os.environ["OPENAI_API_KEY"] = api_key
            elif "anthropic" in default_model:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            elif "openai" in default_model or "gpt" in default_model:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
            elif "gemini" in default_model.lower():
                os.environ.setdefault("GEMINI_API_KEY", api_key)
            elif "zhipu" in default_model or "glm" in default_model or "zai" in default_model:
                os.environ.setdefault("ZHIPUAI_API_KEY", api_key)
            elif "groq" in default_model:
                os.environ.setdefault("GROQ_API_KEY", api_key)
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
    
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
        model = model or self.default_model

        debug = os.getenv("NANOBOT_LLM_DEBUG", "").lower() in ("1", "true", "yes")
        request_id = uuid.uuid4().hex[:8]
        
        # For OpenRouter, prefix model name if not already prefixed
        if self.is_openrouter and not model.startswith("openrouter/"):
            model = f"openrouter/{model}"
        
        # For Zhipu/Z.ai, ensure prefix is present
        # Handle cases like "glm-4.7-flash" -> "zai/glm-4.7-flash"
        if ("glm" in model.lower() or "zhipu" in model.lower()) and not (
            model.startswith("zhipu/") or 
            model.startswith("zai/") or 
            model.startswith("openrouter/")
        ):
            model = f"zai/{model}"
        
        # For custom OpenAI-compatible endpoints, keep the model name as-is.
        # If a hosted_vllm/ prefix is required, the user should include it explicitly.
        
        # For Gemini, ensure gemini/ prefix if not already present
        if "gemini" in model.lower() and not model.startswith("gemini/"):
            model = f"gemini/{model}"
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Ensure LiteLLM uses the provided key for custom endpoints.
        # Some LiteLLM paths default to a placeholder key if api_key isn't explicit.
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Pass api_base directly for custom endpoints (vLLM, etc.)
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        if tools:
            # Some OpenAI-compatible proxies reject advanced JSON Schema keywords.
            removed_keys: set[str] = set()

            def _sanitize(obj: Any) -> Any:
                if isinstance(obj, dict):
                    out = {}
                    for k, v in obj.items():
                        if k in {"default", "minimum", "maximum"}:
                            removed_keys.add(k)
                            continue
                        out[k] = _sanitize(v)
                    return out
                if isinstance(obj, list):
                    return [_sanitize(item) for item in obj]
                return obj

            sanitized_tools = _sanitize(tools)
            kwargs["tools"] = sanitized_tools
            kwargs["tool_choice"] = "auto"

        if self.strip_temperature:
            kwargs.pop("temperature", None)

        if debug:
            logger.info(
                "LLM request: id={} model={} api_base={} is_vllm={} is_openrouter={} messages={} tools={}",
                request_id,
                model,
                self.api_base or "default",
                self.is_vllm,
                self.is_openrouter,
                len(messages),
                bool(tools),
            )
            logger.info(
                "LLM route: id={} force_chat_completions={} strip_temperature={}",
                request_id,
                self.force_chat_completions,
                self.strip_temperature,
            )
            if removed_keys:
                logger.info("LLM tools schema sanitized (removed keys: {})", ",".join(sorted(removed_keys)))
            logger.info(
                "LLM env keys set: OPENAI_API_KEY={} OPENROUTER_API_KEY={} ANTHROPIC_API_KEY={}",
                bool(os.environ.get("OPENAI_API_KEY")),
                bool(os.environ.get("OPENROUTER_API_KEY")),
                bool(os.environ.get("ANTHROPIC_API_KEY")),
            )
            try:
                payload = {k: v for k, v in kwargs.items() if k not in {"api_key", "api_base"}}
                payload_bytes = len(json.dumps(payload, ensure_ascii=True))
                redacted_payload = self._redact_payload(payload)
                redacted_json = json.dumps(redacted_payload, ensure_ascii=True)
                if len(redacted_json) > 4000:
                    redacted_json = f"{redacted_json[:4000]}...(truncated)"
                logger.info(
                    "LLM payload: id={} bytes={} json={}",
                    request_id,
                    payload_bytes,
                    redacted_json,
                )
            except Exception:
                logger.exception("LLM payload log failed: id={}", request_id)

        try:
            if self.force_chat_completions and self.api_base:
                response = await self._chat_completions_httpx(kwargs, request_id=request_id)
                return self._parse_response_dict(response)
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            if isinstance(e, httpx.HTTPStatusError):
                headers = self._select_response_headers(e.response.headers)
                if headers:
                    logger.error(
                        "LLM response headers: id={} headers={}",
                        request_id,
                        headers,
                    )
            if debug:
                logger.exception(
                    "LLM request failed: id={} model={} api_base={} is_vllm={}",
                    request_id,
                    model,
                    self.api_base or "default",
                    self.is_vllm,
                )
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

    async def _chat_completions_httpx(self, kwargs: dict[str, Any], request_id: str) -> dict[str, Any]:
        url = self._build_chat_completions_url(self.api_base or "")
        payload = {k: v for k, v in kwargs.items() if k not in {"api_key", "api_base"}}
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                body = e.response.text or ""
                if len(body) > 2000:
                    body = f"{body[:2000]}...(truncated)"
                headers = self._select_response_headers(e.response.headers)
                logger.error(
                    "Chat completions HTTP error: id={} status={} url={} body={}",
                    request_id,
                    e.response.status_code,
                    e.request.url,
                    body,
                )
                if headers:
                    logger.error(
                        "Chat completions HTTP headers: id={} headers={}",
                        request_id,
                        headers,
                    )
                raise

    @staticmethod
    def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
        def _truncate(value: Any, limit: int = 500) -> Any:
            if value is None:
                return None
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True)
            if len(text) <= limit:
                return text
            return f"{text[:limit]}...(truncated)"

        out: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "messages" and isinstance(value, list):
                messages = []
                for msg in value:
                    if not isinstance(msg, dict):
                        messages.append(_truncate(msg))
                        continue
                    item: dict[str, Any] = {}
                    for field in ("role", "name", "tool_call_id"):
                        if field in msg:
                            item[field] = msg[field]
                    if "content" in msg:
                        item["content"] = _truncate(msg["content"])
                    if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                        calls = []
                        for tc in msg["tool_calls"]:
                            fn = tc.get("function") if isinstance(tc, dict) else None
                            args = fn.get("arguments") if isinstance(fn, dict) else None
                            calls.append({
                                "name": fn.get("name") if isinstance(fn, dict) else None,
                                "arguments_size": len(args) if isinstance(args, str) else len(json.dumps(args, ensure_ascii=True)) if args is not None else 0,
                            })
                        item["tool_calls"] = calls
                    messages.append(item)
                out[key] = messages
                continue

            if key == "tools" and isinstance(value, list):
                tools = []
                for tool in value:
                    if not isinstance(tool, dict):
                        tools.append({"type": type(tool).__name__})
                        continue
                    fn = tool.get("function", {})
                    params = fn.get("parameters") if isinstance(fn, dict) else None
                    props = params.get("properties", {}) if isinstance(params, dict) else {}
                    tools.append({
                        "name": fn.get("name") if isinstance(fn, dict) else None,
                        "params": list(props.keys()),
                    })
                out[key] = tools
                continue

            out[key] = value

        return out

    @staticmethod
    def _select_response_headers(headers: Any) -> dict[str, str]:
        if not headers:
            return {}
        try:
            return dict(headers)
        except Exception:
            return {key: headers.get(key) for key in headers}

    @staticmethod
    def _build_chat_completions_url(api_base: str) -> str:
        base = api_base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _parse_response_dict(self, response: dict[str, Any]) -> LLMResponse:
        try:
            choice = response["choices"][0]
        except (KeyError, IndexError, TypeError):
            return LLMResponse(
                content=None,
                finish_reason="error",
            )

        message = choice.get("message", {})
        content = message.get("content")
        finish_reason = choice.get("finish_reason") or "stop"

        tool_calls: list[ToolCallRequest] = []
        if isinstance(message, dict):
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCallRequest(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args or {},
                ))
            if not tool_calls and message.get("function_call"):
                fn = message.get("function_call") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCallRequest(
                    id="",
                    name=fn.get("name", ""),
                    arguments=args or {},
                ))

        usage = {}
        if isinstance(response.get("usage"), dict):
            usage = response["usage"]

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
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
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
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
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
