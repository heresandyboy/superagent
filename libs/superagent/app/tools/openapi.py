import asyncio
import json
import logging

from langchain.chains.openai_functions.openapi import get_openapi_chain
from langchain_community.tools import BaseTool

logger = logging.getLogger(__name__)


class Openapi(BaseTool):
    name = "API"
    description = "useful for querying an api"
    return_direct = False

    def _run(self, input: str) -> str:
        try:
            openapi_url = self.metadata["openApiUrl"]
            headers = self.metadata.get("headers")
            logger.info(f"Querying OpenAPI: {openapi_url}")
            logger.debug(f"OpenAPI headers: {headers}")

            agent = get_openapi_chain(
                spec=openapi_url, headers=json.loads(
                    headers) if headers else None
            )
            output = agent.run(input)

            logger.info("OpenAPI query completed successfully")
            logger.debug(f"OpenAPI output: {output}")
            return output
        except Exception as e:
            logger.error(f"Error in OpenAPI _run: {e}")
            return str(e)

    async def _arun(self, input: str) -> str:
        try:
            openapi_url = self.metadata["openApiUrl"]
            headers = self.metadata.get("headers")
            logger.info(f"Querying OpenAPI asynchronously: {openapi_url}")
            logger.debug(f"OpenAPI headers: {headers}")

            try:
                agent = get_openapi_chain(
                    spec=openapi_url, headers=json.loads(
                        headers) if headers else None
                )
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(None, agent.run, input)

                logger.info(
                    "Asynchronous OpenAPI query completed successfully")
                logger.debug(f"OpenAPI output: {output}")
            except Exception as e:
                output = str(e)
                logger.error(f"Error in OpenAPI _arun: {e}")

            return output
        except Exception as e:
            logger.error(f"Error in OpenAPI _arun: {e}")
            return str(e)
