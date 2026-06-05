import time
from typing import Any, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agente_edu.config import settings


class Soberano(BaseChatModel):
    base_url: str = settings.soberano_api_base_url
    api_key: str = settings.soberano_api_key
    model: str = settings.soberano_model
    timeout: int = 60
    max_retries: int = 3

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        converted = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            converted.append({"role": role, "content": msg.content})
        return converted

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            **kwargs,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]["content"]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=message))])
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Erro ao chamar o Soberano após {self.max_retries} tentativas: {last_error}")

    @property
    def _llm_type(self) -> str:
        return "soberano-1.1"
