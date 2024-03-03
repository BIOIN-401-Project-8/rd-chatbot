from typing import Any, List, Optional

from llama_index.core.base_retriever import BaseRetriever
from llama_index.node_parser import TextSplitter
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts.base import BasePromptTemplate
from llama_index.query_engine import CitationQueryEngine
from llama_index.query_engine.citation_query_engine import (
    CITATION_QA_TEMPLATE,
    CITATION_REFINE_TEMPLATE,
    DEFAULT_CITATION_CHUNK_OVERLAP,
    DEFAULT_CITATION_CHUNK_SIZE,
)
from llama_index.response_synthesizers import BaseSynthesizer, ResponseMode, get_response_synthesizer
from llama_index.schema import MetadataMode
from llama_index.service_context import ServiceContext


class CustomCitationQueryEngine(CitationQueryEngine):
    @classmethod
    def from_args(
        cls,
        service_context: ServiceContext,
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
    ) -> "CitationQueryEngine":
        """Initialize a CitationQueryEngine object.".

        Args:
            index: (BastGPTIndex): index to use for querying
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
        response_synthesizer = response_synthesizer or get_response_synthesizer(
            service_context=service_context,
            text_qa_template=citation_qa_template,
            refine_template=citation_refine_template,
            response_mode=response_mode,
            use_async=use_async,
            streaming=streaming,
            verbose=verbose,
        )

        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            callback_manager=service_context.callback_manager,
            citation_chunk_size=citation_chunk_size,
            citation_chunk_overlap=citation_chunk_overlap,
            text_splitter=text_splitter,
            node_postprocessors=node_postprocessors,
            metadata_mode=metadata_mode,
        )
