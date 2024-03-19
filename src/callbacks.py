from typing import Any, Dict, List, Optional

from chainlit.element import Text
from chainlit.llama_index.callbacks import LlamaIndexCallbackHandler
from chainlit.step import Step, StepType
from literalai import ChatGeneration, CompletionGeneration, GenerationMessage
from literalai.helper import utc_now
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse

DEFAULT_IGNORE = [
    CBEventType.CHUNKING,
    CBEventType.SYNTHESIZE,
    CBEventType.EMBEDDING,
    CBEventType.NODE_PARSING,
    CBEventType.QUERY,
    CBEventType.TREE,
]


class CustomLlamaIndexCallbackHandler(LlamaIndexCallbackHandler):
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        self._restore_context()
        step_type: StepType = "undefined"
        if event_type == CBEventType.RETRIEVE:
            step_type = "retrieval"
        elif event_type == CBEventType.LLM:
            step_type = "llm"
        else:
            return event_id

        step = Step(
            name=event_type.value,
            type=step_type,
            parent_id=self._get_parent_id(parent_id),
            id=event_id,
            disable_feedback=False,
            show_input=False,
        )
        self.steps[event_id] = step
        step.start = utc_now()
        step.input = payload or {}
        self.context.loop.create_task(step.send())
        return event_id

    def on_event_end(
            self,
            event_type: CBEventType,
            payload: Optional[Dict[str, Any]] = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            """Run when an event ends."""
            step = self.steps.get(event_id, None)

            if payload is None or step is None:
                return

            self._restore_context()

            step.end = utc_now()

            if event_type == CBEventType.RETRIEVE:
                sources = payload.get(EventPayload.NODES)
                if sources:
                    source_refs = "\, ".join(
                        [f"Source {idx}" for idx, _ in enumerate(sources)]
                    )
                    step.elements = [
                        Text(
                            name=f"Source {idx}",
                            content=source.node.get_text() or "Empty node",
                        )
                        for idx, source in enumerate(sources)
                    ]
                    step.output = f"Retrieved the following sources: {source_refs}"
                self.context.loop.create_task(step.update())

            if event_type == CBEventType.LLM:
                formatted_messages = payload.get(
                    EventPayload.MESSAGES
                )  # type: Optional[List[ChatMessage]]
                formatted_prompt = payload.get(EventPayload.PROMPT)
                response = payload.get(EventPayload.RESPONSE)

                if formatted_messages:
                    messages = [
                        GenerationMessage(
                            role=m.role.value, content=m.content or ""  # type: ignore
                        )
                        for m in formatted_messages
                    ]
                else:
                    messages = None

                if isinstance(response, ChatResponse):
                    content = response.message.content or ""
                elif isinstance(response, CompletionResponse):
                    content = response.text
                else:
                    content = ""

                step.output = content

                token_count = self.total_llm_token_count or None

                if messages and isinstance(response, ChatResponse):
                    msg: ChatMessage = response.message
                    step.generation = ChatGeneration(
                        messages=messages,
                        message_completion=GenerationMessage(
                            role=msg.role.value,  # type: ignore
                            content=content,
                        ),
                        token_count=token_count,
                    )
                elif formatted_prompt:
                    step.generation = CompletionGeneration(
                        prompt=formatted_prompt,
                        completion=content,
                        token_count=token_count,
                    )

                self.context.loop.create_task(step.update())

            self.steps.pop(event_id, None)
