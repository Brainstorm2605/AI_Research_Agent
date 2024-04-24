from pydantic import BaseModel, Field
from pydantic.types import StrictStr
from typing import Optional, List, Dict, Any, Union

try:
    from typing import Literal
except ImportError:
    # should have been installed for Python3.7 and above
    from typing_extensions import Literal


class LogProbs(BaseModel, extra="forbid"):
    tokens: List[str] = Field(default_factory=list)
    token_logprobs: List[float] = Field(default_factory=list)
    top_logprobs: Optional[List[Dict[str, float]]] = Field(default_factory=list)
    text_offset: List[int] = Field(default_factory=list)
    # Extension of OpenAI API: also return token ids
    token_ids: Optional[List[int]] = None


class NewLogProbsContentTopLogProbs(BaseModel, extra="forbid"):
    token: str
    logprob: float
    token_id: int
    bytes: List[int] = Field(default_factory=list)


class NewLogProbsContent(BaseModel, extra="forbid"):
    token: str
    logprob: float
    bytes: List[int]
    top_logprobs: List[NewLogProbsContentTopLogProbs] = Field(default_factory=list)
    token_id: int
    text_offset: int


class NewLogProbs(BaseModel, extra="forbid"):
    content: List[NewLogProbsContent] = Field(default_factory=list)


class UsageInfo(BaseModel, extra="forbid"):
    """Usage statistics.

    Attributes:
      prompt_tokens (int): The number of tokens in the prompt.
      total_tokens (int): The total number of tokens used in the request (prompt + completion).
      completion_tokens (Optional[int]): The number of tokens in the generated completion.
    """

    prompt_tokens: int
    total_tokens: int
    completion_tokens: Optional[int] = None


class RawOutput(BaseModel, extra="forbid"):
    """
    Extension of OpenAI that returns low-level interaction of what the model
    sees, including the formatted prompt and function calls
    """

    # Pieces of the prompt (like individual messages) before truncation and
    # concatenation. Depending on prompt_truncate_len some of the messages might
    # be dropped. Contains a mix of strings to be tokenized and individual
    # tokens (if dictated by the conversation template)
    prompt_fragments: List[Union[str, int]]
    # Fully processed prompt as seen by the model
    prompt_token_ids: List[int]
    # Raw completion produced by the model before any tool calls are parsed
    completion: str
    # Log probabilities for the completion. Only populated if logprobs is
    # specified in the request
    completion_logprobs: Optional[NewLogProbs] = None


class Choice(BaseModel, extra="forbid"):
    """A completion choice.

    Attributes:
      index (int): The index of the completion choice.
      text (str): The completion response.
      logprobs (float, optional): The log probabilities of the most likely tokens.
      finish_reason (str): The reason the model stopped generating tokens. This will be "stop" if
        the model hit a natural stop point or a provided stop sequence, or
        "length" if the maximum number of tokens specified in the request was
        reached.
    """

    index: int
    text: str
    logprobs: Optional[Union[LogProbs, NewLogProbs]] = None
    finish_reason: Optional[Literal["stop", "length", "error"]] = None
    raw_output: Optional[RawOutput] = None


class CompletionResponse(BaseModel, extra="forbid"):
    """The response message from a /v1/completions call.

    Attributes:
      id (str): A unique identifier of the response.
      object (str): The object type, which is always "text_completion".
      created (int): The Unix time in seconds when the response was generated.
      choices (List[Choice]): The list of generated completion choices.
    """

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel, extra="forbid"):
    """A streamed completion choice.

    Attributes:
      index (int): The index of the completion choice.
      text (str): The completion response.
      logprobs (float, optional): The log probabilities of the most likely tokens.
      finish_reason (str): The reason the model stopped generating tokens. This will be "stop" if
        the model hit a natural stop point or a provided stop sequence, or
        "length" if the maximum number of tokens specified in the request was
        reached.
    """

    index: int
    text: str
    logprobs: Optional[Union[LogProbs, NewLogProbs]] = None
    finish_reason: Optional[Literal["stop", "length", "error"]] = None


class CompletionStreamResponse(BaseModel, extra="forbid"):
    """The streamed response message from a /v1/completions call.

    Attributes:
      id (str): A unique identifier of the response.
      object (str): The object type, which is always "text_completion".
      created (int): The Unix time in seconds when the response was generated.
      model (str): The model used for the chat completion.
      choices (List[CompletionResponseStreamChoice]):
        The list of streamed completion choices.
    """

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class Model(BaseModel, extra="forbid"):
    """A model deployed to the Fireworks platform.

    Attributes:
      id (str): The model name.
      object (str): The object type, which is always "model".
      created (int): The Unix time in seconds when the model was generated.
      owned_by (str): The organization account owning the model
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ListModelsResponse(BaseModel, extra="forbid"):
    """The response message from a /v1/models call.

    Attributes:
      object (str): The object type, which is always "list".
      data (List[Model]): The list of models.
    """

    object: str = "list"
    data: List[Model]


class ChatCompletionMessageToolCallFunction(BaseModel, extra="forbid"):
    name: Optional[str] = None
    arguments: Optional[str] = None


# TODO: maybe split the function for streaming into a different struct?
class ChatCompletionMessageToolCall(BaseModel, extra="forbid"):
    # TODO: openai requires an index here, will default to zero since we can only
    # return 1 call right now
    index: int = 0
    id: Optional[str] = None
    type: str = "function"
    function: Union[ChatCompletionMessageToolCallFunction, str]


class ChatMessageContentImageURL(BaseModel, extra="forbid"):
    url: str
    detail: Optional[str] = None


class ChatMessageContent(BaseModel, extra="forbid"):
    type: StrictStr
    text: Optional[StrictStr] = None
    image_url: Optional[ChatMessageContentImageURL] = None


class ChatMessage(BaseModel, extra="forbid"):
    """A chat completion message.

    Attributes:
      role (str): The role of the author of this message.
      content (str): The contents of the message.
      tool_calls (list[ChatCompletionMessageToolCall]): used by tools call to communicate the tools call information
      tool_call_id (str): used by tools call to identify which call it was
      name (str): Unused. For OpenAI compatability.
    """

    role: StrictStr
    content: Optional[Union[StrictStr, List[ChatMessageContent]]] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    tool_call_id: Optional[StrictStr] = None
    function: Optional[ChatCompletionMessageToolCallFunction] = None
    name: Optional[StrictStr] = None

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        kwargs.pop("exclude_none", None)
        return super().dict(*args, exclude_none=True, **kwargs)


class ChatCompletionFunction(BaseModel, extra="forbid"):
    name: StrictStr
    description: Optional[StrictStr] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ChatCompletionTool(BaseModel, extra="forbid"):
    type: Literal["function"] = Field(
        ...,
        description="The type of the tool. Currently, only `function` is supported.",
    )
    function: ChatCompletionFunction


class ChatCompletionResponseChoice(BaseModel, extra="forbid"):
    """A chat completion choice generated by a chat model.

    Attributes:
      index (int): The index of the chat completion choice.
      message (ChatMessage): The chat completion message.
      finish_reason (Optional[str]): The reason the model stopped generating tokens. This will be "stop" if
        the model hit a natural stop point or a provided stop sequence, or
        "length" if the maximum number of tokens specified in the request was
        reached.
    """

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    # Extension of OpenAI API
    logprobs: Optional[Union[LogProbs, NewLogProbs]] = None
    raw_output: Optional[RawOutput] = None


class ChatCompletionResponse(BaseModel, extra="forbid"):
    """The response message from a /v1/chat/completions call.

    Attributes:
      id (str): A unique identifier of the response.
      object (str): The object type, which is always "chat.completion".
      created (int): The Unix time in seconds when the response was generated.
      model (str): The model used for the chat completion.
      choices (List[ChatCompletionResponseChoice]): The list of chat completion choices.
      usage (UsageInfo): Usage statistics for the chat completion.
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    # extension of OpenAI API
    usage: Optional[UsageInfo] = None


class DeltaMessage(BaseModel, extra="forbid"):
    """A message delta.

    Attributes:
      role (str): The role of the author of this message.
      content (str): The contents of the chunk message.
    """

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    function: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel, extra="forbid"):
    """A streamed chat completion choice.

    Attributes:
      index (int): The index of the chat completion choice.
      delta (DeltaMessage): The message delta.
      finish_reason (str): The reason the model stopped generating tokens. This will be "stop" if
        the model hit a natural stop point or a provided stop sequence, or
        "length" if the maximum number of tokens specified in the request was
        reached.
    """

    index: int
    delta: DeltaMessage
    finish_reason: Optional[
        Literal["stop", "length", "function_call", "tool_calls"]
    ] = None
    # extension of OpenAI API
    logprobs: Optional[Union[LogProbs, NewLogProbs]] = None


class ChatCompletionStreamResponse(BaseModel, extra="forbid"):
    """The streamed response message from a /v1/chat/completions call.

    Attributes:
      id (str): A unique identifier of the response.
      object (str): The object type, which is always "chat.completion".
      created (int): The Unix time in seconds when the response was generated.
      model (str): The model used for the chat completion.
      choices (List[ChatCompletionResponseStreamChoice]):
        The list of streamed chat completion choices.
    """

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class EmbeddingInfo(BaseModel, extra="forbid"):
    index: int
    embedding: List[float]
    object: str = "embedding"


class EmbeddingResponse(BaseModel, extra="forbid"):
    """The response message from a /v1/embeddings call."""

    data: List[EmbeddingInfo]
    model: str
    object: str = "list"
    usage: UsageInfo


class Document(BaseModel):
    """
    A list of document objects or strings to rerank.
    If a document is provided the text fields is required and all other fields will be preserved in the response.

    The total max chunks (length of documents * max_chunks_per_doc) must be less than 10000.

    We recommend a maximum of 1,000 documents for optimal endpoint performance.
    """

    index: int
    relevance_score: float
    text: Optional[str] = None


class RerankResponse(BaseModel):
    """
    An ordered list of ranked documents
    """

    results: List[Document]
