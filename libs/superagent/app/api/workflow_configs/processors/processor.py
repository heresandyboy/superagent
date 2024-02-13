from app.api.workflow_configs.api.api_manager import ApiManager
from app.api.workflow_configs.processors.base import BaseProcessor
from prisma.enums import AgentType

from .openai import (
    OpenaiDataProcessor,
    OpenaiToolProcessor,
)
from .superagent import (
    SuperagentDataProcessor,
    SuperagentToolProcessor,
)


class Processor:
    def __init__(self, api_user, api_manager: ApiManager):
        self.api_user = api_user
        self.api_manager = api_manager

    def get_data_processor(self, assistant: dict) -> BaseProcessor:
        if assistant.get("type") == AgentType.OPENAI_ASSISTANT:
            return OpenaiDataProcessor(assistant, self.api_manager)
        return SuperagentDataProcessor(assistant, self.api_manager)

    def get_tool_processor(self, assistant: dict) -> BaseProcessor:
        if assistant.get("type") == AgentType.OPENAI_ASSISTANT:
            return OpenaiToolProcessor(assistant, self.api_manager)
        return SuperagentToolProcessor(assistant, self.api_manager)
