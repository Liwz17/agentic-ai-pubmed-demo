from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from openai import OpenAI

from config import MODEL_NAME, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class BaseAgent:
    """
    Shared foundation for chat-based agents in the multi-agent pipeline.

    The class is intentionally lightweight:
    - each agent owns its own OpenAI client
    - each agent keeps its own system prompt and message history
    - tool/function registration is supported as a simple scaffold for later use
    """

    def __init__(
        self,
        system_prompt: str,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.client = client or OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model = model or MODEL_NAME
        self.system_prompt = system_prompt.strip()
        self.messages: List[Dict[str, str]] = []
        self.function_repository: Dict[str, Callable[..., Any]] = {}

        if self.system_prompt:
            self.messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            )

    def add_functions(self, functions: Iterable[Callable[..., Any]]) -> None:
        """
        Register callable tools by function name.

        This is intentionally minimal for now. Later agent implementations can
        decide how to expose these functions to the chat model.
        """
        for func in functions:
            if not callable(func):
                raise TypeError("All registered functions must be callable.")

            name = getattr(func, "__name__", None)
            if not name:
                raise ValueError("Registered functions must have a valid __name__.")

            self.function_repository[name] = func

    def add_message(self, role: str, content: str) -> None:
        """
        Append a chat message to the agent's private memory.
        """
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role: {role}")

        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    def _call_chat_model(
        self,
        user_message: Optional[str] = None,
        temperature: float = 0.0,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        **completion_kwargs: Any,
    ) -> Any:
        """
        Call the chat model once and persist the request/response in memory.

        Notes:
        - Function/tool calling is not executed here yet.
        - Additional completion kwargs can be passed through when needed.
        """
        if user_message:
            self.add_message("user", user_message)

        request_messages = list(self.messages)
        if extra_messages:
            request_messages.extend(extra_messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            temperature=temperature,
            **completion_kwargs,
        )

        assistant_message = response.choices[0].message
        assistant_content = assistant_message.content or ""
        self.add_message("assistant", assistant_content)

        return response

    def _call_chat_model_streaming(
        self,
        user_message: str,
        temperature: float = 0.0,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        **completion_kwargs: Any,
    ):
        """
        Stream a chat completion and store the final assembled assistant text.

        Returns the raw stream iterator so callers can consume chunks directly.
        """
        self.add_message("user", user_message)

        request_messages = list(self.messages)
        if extra_messages:
            request_messages.extend(extra_messages)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            temperature=temperature,
            stream=True,
            **completion_kwargs,
        )

        def _stream_wrapper():
            collected_chunks: List[str] = []

            for chunk in stream:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    collected_chunks.append(text)
                yield chunk

            self.add_message("assistant", "".join(collected_chunks))

        return _stream_wrapper()

    def clear(self) -> None:
        """
        Reset the agent's memory while preserving its system prompt and tools.
        """
        self.messages = []
        if self.system_prompt:
            self.messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            )
