from typing import Any, Dict

from .api import ChatCompletionResponse, ChatCompletionStreamResponse
from .api import ChatMessage
from .base_completion import BaseCompletion
from .error import InvalidRequestError
from .api_client import FireworksClient


class ChatCompletion(BaseCompletion):
    """
    Class for handling chat completions.
    """

    endpoint = "chat/completions"
    response_class = ChatCompletionResponse
    stream_response_class = ChatCompletionStreamResponse

    @classmethod
    def _get_data_key(cls) -> str:
        return "messages"


class ChatCompletionV2(ChatCompletion):
    """
    Class for handling chat completions.
    """

    def __init__(self, client: FireworksClient):
        self._client = client

    def create(
        self,
        model,
        prompt_or_messages=None,
        request_timeout=600,
        stream=False,
        **kwargs,
    ):
        return super().create(
            model,
            prompt_or_messages,
            request_timeout,
            stream=stream,
            client=self._client,
            **self._validate_kwargs(kwargs),
        )

    def acreate(
        self,
        model,
        prompt_or_messages=None,
        request_timeout=600,
        stream=True,
        **kwargs,
    ):
        return super().acreate(
            model,
            prompt_or_messages,
            request_timeout,
            stream=stream,
            client=self._client,
            **self._validate_kwargs(kwargs),
        )

    @classmethod
    def _validate_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "messages" in kwargs:
            messages_as_dict = []
            for message in kwargs["messages"]:
                if isinstance(message, dict):
                    messages_as_dict.append(message)
                elif isinstance(message, ChatMessage):
                    messages_as_dict.append(message.dict())
                else:
                    raise InvalidRequestError(
                        "Chat messages must be a dict or ChatMessage type"
                    )
            kwargs["messages"] = messages_as_dict
        return kwargs
