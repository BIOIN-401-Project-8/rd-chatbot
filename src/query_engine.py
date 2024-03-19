from typing import Any, List, Optional

from llama_index.core import BasePromptTemplate, get_response_synthesizer
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.indices.base import BaseGPTIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import TextSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.query_engine.citation_query_engine import (
    CITATION_QA_TEMPLATE,
    CITATION_REFINE_TEMPLATE,
    DEFAULT_CITATION_CHUNK_OVERLAP,
    DEFAULT_CITATION_CHUNK_SIZE
)
from llama_index.core.response_synthesizers import BaseSynthesizer, ResponseMode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import MetadataMode
from llama_index.core.settings import Settings, callback_manager_from_settings_or_context

from chat_engine.citation_types import CitationChatMode
from chat_engine.citation_condense_plus_context import CitationCondensePlusContextChatEngine

class CustomCitationQueryEngine(CitationQueryEngine):
    @classmethod
    def from_args(
        cls,
        index: BaseGPTIndex | None = None,
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        citation_chunk_size: int = DEFAULT_CITATION_CHUNK_SIZE,
        citation_chunk_overlap: int = DEFAULT_CITATION_CHUNK_OVERLAP,
        text_splitter: Optional[TextSplitter] = None,
        citation_qa_template: BasePromptTemplate = CITATION_QA_TEMPLATE,
        citation_refine_template: BasePromptTemplate = CITATION_REFINE_TEMPLATE,
        retriever: Optional[BaseRetriever] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        use_async: bool = False,
        streaming: bool = False,
        verbose: bool = False,
        # class-specific args
        metadata_mode: MetadataMode = MetadataMode.NONE,
        **kwargs: Any,
    ) -> "CustomCitationQueryEngine":
        """Initialize a CitationQueryEngine object.".

        Args:
            index: (BastGPTIndex): index to use for querying
            llm: (Optional[LLM]): LLM object to use for response generation.
            citation_chunk_size (int):
                Size of citation chunks, default=512. Useful for controlling
                granularity of sources.
            citation_chunk_overlap (int): Overlap of citation nodes, default=20.
            text_splitter (Optional[TextSplitter]):
                A text splitter for creating citation source nodes. Default is
                a SentenceSplitter.
            citation_qa_template (BasePromptTemplate): Template for initial citation QA
            citation_refine_template (BasePromptTemplate):
                Template for citation refinement.
            retriever (BaseRetriever): A retriever object.
            service_context (Optional[ServiceContext]): A ServiceContext object.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
                node postprocessors.
            verbose (bool): Whether to print out debug info.
            response_mode (ResponseMode): A ResponseMode object.
            use_async (bool): Whether to use async.
            streaming (bool): Whether to use streaming.
            optimizer (Optional[BaseTokenUsageOptimizer]): A BaseTokenUsageOptimizer
                object.

        """
        retriever = retriever or index.as_retriever(**kwargs)

        response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=llm,
            service_context=index.service_context if index else None,
            text_qa_template=citation_qa_template,
            refine_template=citation_refine_template,
            response_mode=response_mode,
            use_async=use_async,
            streaming=streaming,
            verbose=verbose,
        )

        return cls(
            retriever=retriever,
            llm=llm,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager_from_settings_or_context(
                Settings, index.service_context if index else None
            ),
            citation_chunk_size=citation_chunk_size,
            citation_chunk_overlap=citation_chunk_overlap,
            text_splitter=text_splitter,
            node_postprocessors=node_postprocessors,
            metadata_mode=metadata_mode,
        )

    def as_chat_engine(
        self,
        chat_mode: ChatMode = ChatMode.BEST,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> BaseChatEngine:
        # llama_index.core.indices.base.BaseIndex
        # resolve chat mode
        if chat_mode in [ChatMode.REACT, ChatMode.OPENAI, ChatMode.BEST]:
            # use an agent with query engine tool in these chat modes
            # NOTE: lazy import
            from llama_index.core.agent import AgentRunner
            from llama_index.core.tools.query_engine import QueryEngineTool

            # convert query engine to tool
            query_engine_tool = QueryEngineTool.from_defaults(query_engine=self)

            return AgentRunner.from_llm(
                tools=[query_engine_tool],
                llm=llm,
                **kwargs,
            )

        if chat_mode == ChatMode.CONDENSE_QUESTION:
            # NOTE: lazy import
            from llama_index.core.chat_engine import CondenseQuestionChatEngine

            return CondenseQuestionChatEngine.from_defaults(
                query_engine=self,
                llm=llm,
                **kwargs,
            )

        elif chat_mode == ChatMode.CONTEXT:
            from llama_index.core.chat_engine import ContextChatEngine

            return ContextChatEngine.from_defaults(
                retriever=self.retriever,
                llm=llm,
                **kwargs,
            )

        elif chat_mode == ChatMode.CONDENSE_PLUS_CONTEXT:
            from llama_index.core.chat_engine import CondensePlusContextChatEngine

            return CondensePlusContextChatEngine.from_defaults(
                retriever=self.retriever,
                llm=llm,
                **kwargs,
            )

        elif chat_mode == ChatMode.SIMPLE:
            from llama_index.core.chat_engine import SimpleChatEngine

            return SimpleChatEngine.from_defaults(
                llm=llm,
                **kwargs,
            )

        elif chat_mode == CitationChatMode.CONDENSE_QUESTION:
            from chat_engine.citation_condense_question import CitationCondenseQuestionChatEngine

            return CitationCondenseQuestionChatEngine.from_defaults(
                query_engine=self,
                llm=llm,
                **kwargs,
            )

        elif chat_mode == CitationChatMode.CONTEXT:
            from chat_engine.citation_context import CitationContextChatEngine

            return CitationContextChatEngine.from_defaults(
                retriever=self.retriever,
                llm=llm,
                **kwargs,
            )

        elif chat_mode == CitationChatMode.CONDENSE_PLUS_CONTEXT:
            from chat_engine.citation_condense_plus_context import CitationCondensePlusContextChatEngine

            return CitationCondensePlusContextChatEngine.from_defaults(
                retriever=self.retriever,
                llm=llm,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown chat mode: {chat_mode}")
