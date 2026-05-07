"""
cap_eval/llm_client/openai_client.py

A thin wrapper around the OpenAI Python SDK (v1+) that
adds exponential-backoff retry logic, unified synchronous
and asynchronous interfaces, and optional token usage stats.
"""

import os
import backoff
import json
import instructor
from openai import OpenAI, AsyncOpenAI
from openai import (
    OpenAIError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
)
import logging
from pydantic import BaseModel

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _log_backoff(details):
    """Log retry attempts triggered by backoff."""
    exc = details.get("exception")
    tries = details.get("tries")
    wait = details.get("wait")
    target = details.get("target")
    target_name = getattr(target, "__name__", str(target))
    kwargs = details.get("kwargs") or {}
    model = kwargs.get("model")
    if exc is not None:
        logger.warning(
            "OpenAI retry #%s after %.1fs in %s (model=%s) due to %s: %s",
            tries,
            wait or 0,
            target_name,
            model,
            type(exc).__name__,
            exc,
        )
    else:
        logger.warning(
            "OpenAI retry #%s after %.1fs in %s (model=%s, no exception info)",
            tries,
            wait or 0,
            target_name,
            model,
        )


def _log_giveup(details):
    exc = details.get("exception")
    target = details.get("target")
    target_name = getattr(target, "__name__", str(target))
    kwargs = details.get("kwargs") or {}
    model = kwargs.get("model")
    if exc is not None:
        logger.error(
            "OpenAI retries exhausted in %s (model=%s) due to %s: %s",
            target_name,
            model,
            type(exc).__name__,
            exc,
        )
    else:
        logger.error(
            "OpenAI retries exhausted in %s (model=%s, no exception info)",
            target_name,
            model,
        )


# --------------------------------------------------------------------------- #
# Retry helpers                                                               #
# --------------------------------------------------------------------------- #


@backoff.on_exception(
    backoff.expo,
    (OpenAIError, APIConnectionError, RateLimitError, InternalServerError, APITimeoutError),
    on_backoff=_log_backoff,
    on_giveup=_log_giveup,
)
def completion_with_backoff(client: OpenAI, **kwargs):
    """
    Synchronous completion request with exponential-backoff retry.
    """
    if "response_format" in kwargs:
        return client.beta.chat.completions.parse(**kwargs)
    return client.chat.completions.create(**kwargs)


@backoff.on_exception(
    backoff.expo,
    (OpenAIError, APIConnectionError, RateLimitError, InternalServerError, APITimeoutError),
    on_backoff=_log_backoff,
    on_giveup=_log_giveup,
)
async def acompletion_with_backoff(client: AsyncOpenAI, **kwargs):
    """
    Asynchronous completion request with exponential-backoff retry.
    """
    if "response_model" in kwargs:
        return await client.chat.completions.create_parsed(**kwargs)
    if "response_format" in kwargs:
        return await client.beta.chat.completions.parse(**kwargs)
    return await client.chat.completions.create(**kwargs)


# --------------------------------------------------------------------------- #
# Synchronous client                                                          #
# --------------------------------------------------------------------------- #


class OpenAIClient:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def response(self, count_token: bool = False, **kwargs):
        # <--- 新增改动: 强制覆盖所有同步调用的模型名称
        kwargs['model'] = 'gpt-4o-mini'

        response = completion_with_backoff(self.client, **kwargs)
        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        if "response_format" in kwargs:
            return (response.choices[0].message.parsed, tokens) if count_token else response.choices[0].message.parsed
        return (response.choices[0].message.content, tokens) if count_token else response.choices[0].message.content


# --------------------------------------------------------------------------- #
# Asynchronous client                                                         #
# --------------------------------------------------------------------------- #

class AsyncOpenAIClient:
    def __init__(self, api_key: str = None, base_url: str = None) -> None:
        resolved_api_key = "<REMOVED_SECRET>"
        resolved_base_url = "https://litellm.fellou.ai"
        aclient = AsyncOpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self.client = instructor.patch(aclient)

    async def response(self, count_token: bool = False, **kwargs):
        """
        Final, corrected version of the async wrapper.
        """
        # (如果您想继续使用 claude，请保留这行；如果想换 gpt-4o，请修改它)
        kwargs['model'] = 'openai/gpt-5' 
        
        response = await acompletion_with_backoff(self.client, **kwargs)

        # 关键修正：不再使用 isinstance，而是检查调用意图
        if "response_model" in kwargs:
            # `response` 是 ParsedChatCompletion 容器，我们需要里面的 .parsed 对象
            parsed_object = response.choices[0].message.parsed
            tokens = {"input_tokens": None, "output_tokens": None}
            return (parsed_object, tokens) if count_token else parsed_object
        
        # 对于非结构化请求，逻辑保持不变
        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        if "response_format" in kwargs:
            return (response.choices[0].message.parsed, tokens) if count_token else response.choices[0].message.parsed
        return (response.choices[0].message.content, tokens) if count_token else response.choices[0].message.content