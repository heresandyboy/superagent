from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, List, Literal, Tuple, Union, cast

from decouple import config
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import LLMResult
from langfuse import Langfuse
from litellm import cost_per_token, token_counter

logger = logging.getLogger(__name__)


class CustomAsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[str]

    done: asyncio.Event

    TIMEOUT_SECONDS = 60

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=5)
        self.done = asyncio.Event()
        logger.debug("CustomAsyncIteratorCallbackHandler initialized")

    async def on_chat_model_start(
        self,
        *args: Any,  # noqa
        **kwargs: Any,  # noqa
    ) -> None:
        """Run when LLM starts running."""
        logger.debug(
            "on_chat_model_start called with args: %s, kwargs: %s", args, kwargs)
        pass

    async def on_llm_start(self) -> None:
        logger.debug("on_llm_start called")
        # If two calls are made in a row, this resets the state
        self.done.clear()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # noqa
        # logger.debug(
        #     "on_llm_new_token called with token: %s, kwargs: %s", token, kwargs)
        try:
            if token is not None and token != "":
                has_put = False
                while not has_put:
                    try:
                        await self.queue.put(token)
                        has_put = True
                        # logger.debug("Token added to queue: %s", token)
                    except asyncio.QueueFull:
                        logger.debug("Queue is full, waiting to add token")
                        continue
        except Exception as e:
            logger.error(f"Error in on_llm_new_token: {e}")

    async def on_llm_end(self, response, **kwargs: Any) -> None:  # noqa
        logger.debug(
            "on_llm_end called with response: %s, kwargs: %s", response, kwargs)
        try:
            # TODO:
            # This should be removed when Langchain has merged
            # https://github.com/langchain-ai/langchain/pull/9536
            for gen_list in response.generations:
                for gen in gen_list:
                    if gen.message.content != "":
                        self.done.set()
                        logger.debug("LLM response received, set done event")
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")

    async def on_llm_error(self, *args: Any, **kwargs: Any) -> None:  # noqa
        logger.error("LLM error occurred")
        self.done.set()

    async def aiter(self) -> AsyncIterator[str]:
        logger.debug("aiter called")
        try:
            while not self.queue.empty() or not self.done.is_set():
                # logger.debug("Waiting for token or done event")
                done, pending = await asyncio.wait(
                    [
                        asyncio.ensure_future(self.queue.get()),
                        asyncio.ensure_future(self.done.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self.TIMEOUT_SECONDS,
                )
                if not done:
                    logger.warning(
                        f"{self.TIMEOUT_SECONDS} seconds of timeout reached")
                    self.done.set()
                    break

                for future in pending:
                    future.cancel()

                token_or_done = cast(
                    Union[str, Literal[True]], done.pop().result())

                if token_or_done is True:
                    logger.debug("Done event received, continuing")
                    continue

                # logger.debug("Token yielded: %s", token_or_done)
                yield token_or_done
        except Exception as e:
            logger.error(f"Error in aiter: {e}")


class CostCalcAsyncHandler(AsyncCallbackHandler):
    """Callback handler that calculates the cost of the prompt and completion."""

    def __init__(self, model):
        self.model = model
        self.prompt: str = ""
        self.completion: str = ""
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.prompt_tokens_cost_usd: float = 0.0
        self.completion_tokens_cost_usd: float = 0.0
        logger.debug("CostCalcAsyncHandler initialized with model: %s", model)

    def on_llm_start(self, _, prompts: List[str], **kwargs: Any) -> None:  # noqa
        logger.debug(
            "on_llm_start called with prompts: %s, kwargs: %s", prompts, kwargs)
        try:
            self.prompt = prompts[0]
            logger.info("Prompt received: %s", self.prompt)
        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}")

    def on_llm_end(self, llm_result: LLMResult, **kwargs: Any) -> None:  # noqa
        logger.debug(
            "on_llm_end called with llm_result: %s, kwargs: %s", llm_result, kwargs)
        try:
            self.completion = llm_result.generations[0][0].message.content
            logger.info("Completion received: %s", self.completion)
            completion_tokens = self._calculate_tokens(self.completion)
            prompt_tokens = self._calculate_tokens(self.prompt)

            (
                prompt_tokens_cost_usd,
                completion_tokens_cost_usd,
            ) = self._calculate_cost_per_token(prompt_tokens, completion_tokens)

            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.prompt_tokens_cost_usd = prompt_tokens_cost_usd
            self.completion_tokens_cost_usd = completion_tokens_cost_usd
            logger.info("Prompt tokens: %d, Completion tokens: %d, Prompt cost: $%.4f, Completion cost: $%.4f",
                        prompt_tokens, completion_tokens, prompt_tokens_cost_usd, completion_tokens_cost_usd)
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")

    def _calculate_tokens(self, text: str) -> int:
        logger.debug("_calculate_tokens called with text: %s", text)
        try:
            tokens = token_counter(model=self.model, text=text)
            logger.debug("Tokens calculated: %d", tokens)
            return tokens
        except Exception as e:
            logger.error(f"Error calculating tokens: {e}")
            return 0

    def _calculate_cost_per_token(
        self, prompt_tokens: int, completion_tokens: int
    ) -> Tuple[float, float]:
        logger.debug("_calculate_cost_per_token called with prompt_tokens: %d, completion_tokens: %d",
                     prompt_tokens, completion_tokens)
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            logger.debug("Prompt cost: $%.4f, Completion cost: $%.4f",
                         prompt_cost, completion_cost)
            return prompt_cost, completion_cost
        except Exception as e:
            logger.error(f"Error calculating cost per token: {e}")
            return 0.0, 0.0


def get_session_tracker_handler(
    workflow_id,
    agent_id,
    session_id,
    user_id,
):
    logger.debug("get_session_tracker_handler called with workflow_id: %s, agent_id: %s, session_id: %s, user_id: %s",
                 workflow_id, agent_id, session_id, user_id)
    try:
        langfuse_secret_key = config("LANGFUSE_SECRET_KEY", "")
        langfuse_public_key = config("LANGFUSE_PUBLIC_KEY", "")
        langfuse_host = config("LANGFUSE_HOST", "https://cloud.langfuse.com")
        langfuse_handler = None
        if langfuse_public_key and langfuse_secret_key:
            logger.info("Initializing Langfuse session tracker")
            langfuse = Langfuse(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                host=langfuse_host,
                sdk_integration="Superagent",
            )
            trace = langfuse.trace(
                id=session_id,
                name="Workflow",
                tags=[agent_id],
                metadata={"agentId": agent_id, "workflowId": workflow_id},
                user_id=user_id,
                session_id=workflow_id,
            )
            langfuse_handler = trace.get_langchain_handler()
            logger.info("Langfuse session tracker initialized")
            return langfuse_handler
    except Exception as e:
        logger.error(f"Error initializing Langfuse session tracker: {e}")

    logger.warning("Langfuse session tracker not initialized")
    return None
