import os
import time
from dataclasses import dataclass
from typing import Any, Type
from pydantic import BaseModel

import openai
from openai import AsyncOpenAI, OpenAI


@dataclass
class Response:
    content: str # main content
    raw: Any # full raw response
    parsed: Any | None = None # Parsed JSON/dict if using `response_format`
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    duration: float = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def tool_calls(self):
        pass

class Client:
    """
    Minimal wrapper of LLM client

    Auto generate two clients: client() and aclient() for usage
    """

    # TODO: add deepseek and groqcloud support
    # OpenAI-compatible providers
    PROVIDERS = {
        "openai": None,
        "deepseek": "https://api.deepseek.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
        "local": "http://localhost:8000/v1",
    }

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "openai",
        model: str = "o4-mini-2025-04-16",
        base_url: str | None = None,
        temperature: float = 0.7,
        timeout: float | None = None,
        max_retries: int = 3,
        api_max_retries: int = 1,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        if not self.api_key and provider != "local":
            raise ValueError(
                f"No API key found. Set {provider.upper()}_API_KEY or enter api_key"
            )

        # base URL: explicit > provider preset
        self.base_url = base_url or self.PROVIDERS.get(provider)

        # defaults
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        # create clients
        client_args = {
            "api_key": self.api_key,
            "timeout": timeout,
            "max_retries": api_max_retries,
        }
        if self.base_url:
            client_args["base_url"] = self.base_url

        self.client = OpenAI(**client_args)
        self.async_client = AsyncOpenAI(**client_args)

    #TODO: add tools calling later
    def _create(self, messages, instructions=None, tools=None):
        """
        Plain, non-streaming call.
        - Use when you want free-form text or JSON but don’t require strict schema validation.
        - `tools` can be passed for function calling (optional).
        """
        return self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=messages,
            temperature=self.temperature,
            timeout=self.timeout,
        )
    
    def _parse(self, messages, instructions=None, text_format=None):
        """
        Plain, non-streaming call.
        - Use when you want free-form text or JSON but don’t require strict schema validation.
        - `tools` can be passed for function calling (optional).
        """
        return self.client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=messages,
            text_format=text_format,
            temperature=self.temperature,
            timeout=self.timeout
        )
    
    def respond(
        self,
        messages: str | list[dict],
        instruction: str | None = None,
        text_format: Type[BaseModel] | None = None,
        stream: bool = False,
        tools: list | None = None,
        strict: bool = False,
    ):
        """
        Entry point for all LLM calls.
        - `schema`: If provided, forces structured output matching a Pydantic model.
        - `stream`: If True, yields results incrementally instead of returning all at once.
        - `tools`: Optional list of tool definitions (JSON Schema) for function-calling style output.
        - `strict`: If True, enforce exact schema compliance when using tools.
        """
        last_error = None
        response = None

        start_time = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                if stream and text_format:
                    pass
                elif stream:
                    pass
                elif text_format:
                    response = self.client.responses.parse(
                        model=self.model,
                        instructions=instruction,
                        input=messages,
                        text_format=text_format,
                    )
                    break
                # success response
                else:
                    response = self.client.responses.create(
                        model=self.model,
                        instructions=instruction,
                        input=messages,
                    )
                    break
            except openai.RateLimitError as e:
                wait = min(2 ** attempt, 10)  # Cap at 10s
                # logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                last_error = e
                
            except openai.APITimeoutError as e:
                if attempt < self.max_retries - 1:
                    # logger.warning(f"Timeout, retrying ({attempt+1}/{self.max_retries})...")
                    time.sleep(1)
                last_error = e
                
            except Exception as e:
                # logger.error(f"API error: {e}")
                raise
        elapsed_time = time.perf_counter() - start_time
        
        if response is None:
            raise last_error or Exception("Max retries reached without a response")
        
        return self._make_response(response, elapsed_time, attempt)

    #TODO: add async support
    def async_respond():
        pass

    def _make_response(self, api_response, elapsed, retries) -> Response:
        content = None
        parsed_out = None
        
        if hasattr(api_response, "output_text"):
            content = api_response.output_text
        try:
            if hasattr(api_response, "output_parsed"):
                parsed_out = api_response.output_parsed
        except Exception as e:
            print(e)
            pass
        
        try:
            tokens_in = api_response.usage.input_tokens
        except:
            tokens_in = 0

        try:
            tokens_out = api_response.usage.output_tokens
        except:
            tokens_out = 0
        
        model = api_response.model

        return Response(
            content=content,
            raw=api_response,
            parsed=parsed_out,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration=elapsed,
            retries=retries,
        )
