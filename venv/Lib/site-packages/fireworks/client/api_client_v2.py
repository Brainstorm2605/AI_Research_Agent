import httpx

from .api_client import FireworksClient as FireworksClientV1
from .chat import Chat
from .completion import CompletionV2
from .embedding import EmbeddingV1
from .rerank import RerankV1
from typing import Union
from .image import ImageInference


class BaseFireworks:
    _organization: str
    _max_retries: int
    _client_v1: FireworksClientV1
    chat: Chat

    def __init__(
        self,
        *,
        api_key: Union[str, None] = None,
        base_url: Union[str, httpx.URL, None] = None,
        timeout: int = 600,
        account: str = "fireworks",
    ):
        self._client_v1 = FireworksClientV1(
            api_key=api_key, base_url=base_url, request_timeout=timeout
        )
        self._image_client_v1 = ImageInference(
            model=None,
            account=account,
            request_timeout=timeout,
            api_key=api_key,
            base_url=base_url,
        )

        self.completion = CompletionV2(self._client_v1)
        self.chat = Chat(self._client_v1)
        self.embeddings = EmbeddingV1(self._client_v1)
        self.rerank = RerankV1(self._client_v1)


class Fireworks(BaseFireworks):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        return self


class AsyncFireworks(BaseFireworks):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        await self._client_v1.aclose()
        await self._image_client_v1.aclose()
