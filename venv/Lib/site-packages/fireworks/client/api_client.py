import json
import httpx
from typing import Dict, Any, Optional, Generator, Union, AsyncGenerator
from httpx_sse import connect_sse, aconnect_sse
import fireworks.client
from .error import (
    AuthenticationError,
    BadGatewayError,
    InternalServerError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
    ServiceUnavailableError,
)


# Helper functions for api key and base url prevent cyclic dependencies
def default_api_key() -> str:
    if fireworks.client.api_key is not None:
        return fireworks.client.api_key
    else:
        raise ValueError(
            "No API key provided. You can set your API key in code using 'fireworks.client.api_key = <API-KEY>', or you can set the environment variable FIREWORKS_API_KEY=<API-KEY>)."
        )


def default_base_url() -> str:
    return fireworks.client.base_url


class FireworksClient:
    """
    Fireworks client class to help with request handling for
    - get
    - post
      - with & without async
      - with & without streaming
    """

    def __init__(
        self,
        request_timeout=600,
        *,
        api_key: Union[str, None] = None,
        base_url: Union[str, httpx.URL, None] = None,
        **kwargs,
    ) -> None:
        """Initializes the Fireworks client.

        Args:
          request_timeout (int): A request timeout in seconds.
        """
        if "request_timeout" in kwargs:
            request_timeout = kwargs["request_timeout"]
        self.api_key = api_key or default_api_key()
        if not self.api_key:
            raise AuthenticationError(
                "No API key provided. You can set your API key in code using 'fireworks.client.api_key = <API-KEY>', or you can set the environment variable FIREWORKS_API_KEY=<API-KEY>)."
            )
        self.base_url = base_url or default_base_url()
        self.request_timeout = request_timeout
        self.client_kwargs = kwargs

        self._client: httpx.Client = self.create_client()
        self._async_client: httpx.AsyncClient = self.create_async_client()

    def _raise_for_status(self, resp):
        # Function to get error message or default to response code name
        def get_error_message(response, error_type: str = "invalid_request_error"):
            try:
                # Try to return the JSON body
                return response.json()
            except json.decoder.JSONDecodeError:
                # If JSON parsing fails, return the HTTP status code name
                return {
                    "error": {
                        "object": "error",
                        "type": error_type,
                        "message": response.reason_phrase,
                    }
                }

        if resp.status_code == 400:
            raise InvalidRequestError(get_error_message(resp))
        elif resp.status_code == 401:
            raise AuthenticationError(get_error_message(resp))
        elif resp.status_code == 403:
            raise PermissionError(get_error_message(resp))
        elif resp.status_code == 404:
            raise InvalidRequestError(get_error_message(resp))
        elif resp.status_code == 429:
            raise RateLimitError(get_error_message(resp))
        elif resp.status_code == 500:
            raise InternalServerError(get_error_message(resp, "internal_server_error"))
        elif resp.status_code == 502:
            raise BadGatewayError(get_error_message(resp, "internal_server_error"))
        elif resp.status_code == 503:
            raise ServiceUnavailableError(
                get_error_message(resp, "internal_server_error")
            )
        else:
            resp.raise_for_status()

    async def _async_error_handling(self, resp):
        if resp.is_error:
            await resp.aread()
        self._raise_for_status(resp)

    def _error_handling(self, resp):
        if resp.is_error:
            resp.read()
        self._raise_for_status(resp)

    def _get_request(self, url: str) -> Dict[str, Any]:
        resp = self._client.get(url)
        self._error_handling(resp)
        return resp.json()

    def post_request_streaming(
        self, url: str, data: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:
        with connect_sse(self._client, url=url, method="POST", json=data) as event_source:
            self._error_handling(event_source.response)
            for sse in event_source.iter_sse():
                yield sse.data

    def post_request_non_streaming(
        self, url: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        response = self._client.post(url, json=data)
        self._error_handling(response)
        return response.json()

    async def post_request_async_streaming(
        self, url: str, data: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        async with aconnect_sse(
            self._async_client, url=url, method="POST", json=data
        ) as event_source:
            await self._async_error_handling(event_source.response)
            async for sse in event_source.aiter_sse():
                yield sse.data

    async def post_request_async_non_streaming(
        self, url: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        response = await self._async_client.post(url, json=data)
        await self._async_error_handling(response)
        return response.json()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        await self._async_client.aclose()

    def create_client(self) -> httpx.Client:
        return httpx.Client(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.request_timeout,
            **self.client_kwargs,
        )

    def create_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.request_timeout,
            **self.client_kwargs,
        )
