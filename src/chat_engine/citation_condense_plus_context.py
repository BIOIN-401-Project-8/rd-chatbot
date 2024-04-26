import asyncio
import logging
from threading import Thread
from typing import List, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
from llama_index.core.utilities.token_counting import TokenCounter

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_PROMPT_TEMPLATE = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is provides an answer based solely on the provided sources. The
  assistant cites the appropriate source(s) using their corresponding numbers.
  Every answer should include at least one source citation. Only cite a source
  when you are explicitly referencing it. If the assistant does not know the
  answer to a question, it truthfully says it does not know.

  Here are the relevant sources for the context:

  {context_str}

  Instruction: Based on the above sources, provide a detailed answer with
  sources for the user question below. Answer "I'm sorry, I don't know" if not present in the
  document.
  """

DEFAULT_CONDENSE_PROMPT_TEMPLATE = """
  Given the following conversation between a user and an AI assistant and a follow up question from user,
  rephrase the follow up question to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:"""


class CitationCondensePlusContextChatEngine(CondensePlusContextChatEngine):
    """Condensed Conversation & Context Chat Engine.

    First condense a conversation and latest user message to a standalone question
    Then build a context for the standalone question from a retriever,
    Then pass the context along with prompt and user message to LLM to generate a response.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        context_prompt: Optional[str] = None,
        condense_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._context_prompt_template = context_prompt or DEFAULT_CONTEXT_PROMPT_TEMPLATE
        condense_prompt_str = condense_prompt or DEFAULT_CONDENSE_PROMPT_TEMPLATE
        self._condense_prompt_template = PromptTemplate(condense_prompt_str)
        self._system_prompt = system_prompt
        self._skip_condense = skip_condense
        self._node_postprocessors = node_postprocessors or []
        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

        self._token_counter = TokenCounter()
        self._verbose = verbose
        self._nodes = []


    def _create_citation_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Modify retrieved nodes to be granular sources."""
        # Inspired by llama_index.core.query_engine.citation_query_engine.CitationQueryEngine
        new_nodes: List[NodeWithScore] = []
        for node in nodes:
            new_node = NodeWithScore(
                node=TextNode.parse_obj(node.node), score=node.score
            )
            new_node.node.text = f"Source {len(self._nodes) + 1}: {node.text}"
            new_nodes.append(new_node)
            self._nodes.append(new_node)
        return new_nodes

    def _retrieve_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Build context for a message from retriever."""
        nodes = self._retriever.retrieve(message)
        nodes = self._create_citation_nodes(nodes)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=QueryBundle(message))

        context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
        return context_str, nodes

    async def _aretrieve_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Build context for a message from retriever."""
        nodes = await self._retriever.aretrieve(message)
        nodes = self._create_citation_nodes(nodes)
        context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
        return context_str, nodes

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_c3(
            message, chat_history
        )

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = self._llm.chat(chat_messages)
        assistant_message = chat_response.message
        self._memory.put(assistant_message)

        return AgentChatResponse(
            response=str(assistant_message.content),
            sources=[context_source],
            source_nodes=self._nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_c3(
            message, chat_history
        )

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(chat_messages),
            sources=[context_source],
            source_nodes=self._nodes,
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_messages, context_source, context_nodes = await self._arun_c3(
            message, chat_history
        )

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = await self._llm.achat(chat_messages)
        assistant_message = chat_response.message
        self._memory.put(assistant_message)

        return AgentChatResponse(
            response=str(assistant_message.content),
            sources=[context_source],
            source_nodes=self._nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = await self._arun_c3(
            message, chat_history
        )

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(chat_messages),
            sources=[context_source],
            source_nodes=self._nodes,
        )
        thread = Thread(
            target=lambda x: asyncio.run(chat_response.awrite_response_to_history(x)),
            args=(self._memory,),
        )
        thread.start()

        return chat_response

    def reset(self) -> None:
        self._nodes = []
        super().reset()
