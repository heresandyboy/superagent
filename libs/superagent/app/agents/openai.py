import asyncio
import logging

from langchain.agents import AgentExecutor
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.schema.messages import AIMessage
from langchain.schema.output import ChatGeneration, LLMResult

from app.agents.base import AgentBase

logger = logging.getLogger(__name__)


class OpenAiAssistant(AgentBase):
    async def get_agent(self):
        try:
            assistant_id = self.agent_config.metadata.get("id")
            agent = OpenAIAssistantRunnable(
                assistant_id=assistant_id, as_agent=True)
            enable_streaming = self.enable_streaming

            logger.info(
                f"OpenAiAssistant initialized with assistant_id: {assistant_id}")

            class CustomAgentExecutor(AgentExecutor):
                async def ainvoke(self, *args, **kwargs):
                    try:
                        res = await super().ainvoke(*args, **kwargs)
                        logger.debug(f"AgentExecutor result: {res}")

                        if enable_streaming:
                            output = res.get("output").split(" ")
                            # TODO: find a better way to get the streaming callback
                            streaming = kwargs["config"]["callbacks"][0]
                            await streaming.on_llm_start()

                            tasks = []

                            for token in output:
                                task = streaming.on_llm_new_token(token + " ")
                                tasks.append(task)

                            await asyncio.gather(*tasks)

                            await streaming.on_llm_end(
                                response=LLMResult(
                                    generations=[
                                        [
                                            ChatGeneration(
                                                message=AIMessage(
                                                    content=res.get("output"),
                                                )
                                            )
                                        ]
                                    ],
                                )
                            )

                        return res
                    except Exception as e:
                        logger.error(
                            f"Error in CustomAgentExecutor.ainvoke: {e}")
                        raise e

            agent_executor = CustomAgentExecutor(agent=agent, tools=[])

            logger.info("CustomAgentExecutor initialized")
            return agent_executor
        except Exception as e:
            logger.error(f"Error initializing OpenAiAssistant: {e}")
            raise e
