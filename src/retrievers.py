import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import faiss
from llama_index.core import BasePromptTemplate, QueryBundle, ServiceContext, Settings, StorageContext, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.knowledge_graph.retrievers import REL_TEXT_LIMIT
from llama_index.core.llms.llm import LLM
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

logger = logging.getLogger(__name__)


class KG_RAG_KnowledgeGraphRAGRetriever(KnowledgeGraphRAGRetriever):
    def __init__(
        self,
        storage_context: Optional[StorageContext] = None,
        llm: Optional[LLM] = None,
        entity_extract_fn: Optional[Callable] = None,
        entity_extract_template: Optional[BasePromptTemplate] = None,
        entity_extract_policy: Optional[str] = "union",
        synonym_expand_fn: Optional[Callable] = None,
        synonym_expand_template: Optional[BasePromptTemplate] = None,
        synonym_expand_policy: Optional[str] = "union",
        max_entities: int = 5,
        max_synonyms: int = 5,
        retriever_mode: Optional[str] = "keyword",
        with_nl2graphquery: bool = False,
        graph_traversal_depth: int = 2,
        max_knowledge_sequence: int = REL_TEXT_LIMIT,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        # deprecated
        service_context: Optional[ServiceContext] = None,
        similarity_top_k: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the retriever."""
        super().__init__(
            storage_context=storage_context,
            llm=llm,
            entity_extract_fn=entity_extract_fn,
            entity_extract_template=entity_extract_template,
            entity_extract_policy=entity_extract_policy,
            synonym_expand_fn=synonym_expand_fn,
            synonym_expand_template=synonym_expand_template,
            synonym_expand_policy=synonym_expand_policy,
            max_entities=max_entities,
            max_synonyms=max_synonyms,
            retriever_mode=retriever_mode,
            with_nl2graphquery=with_nl2graphquery,
            graph_traversal_depth=graph_traversal_depth,
            max_knowledge_sequence=max_knowledge_sequence,
            verbose=verbose,
            callback_manager=callback_manager,
            service_context=service_context,
            **kwargs,
        )
        self._similarity_top_k = similarity_top_k
        self._verbose = verbose

    def _build_nodes(
        self, knowledge_sequence: List[str], rel_map: Optional[Dict[Any, Any]] = None, query_bundle: QueryBundle = None
    ) -> List[NodeWithScore]:
        """Build nodes from knowledge sequence."""
        if len(knowledge_sequence) == 0:
            logger.info("> No knowledge sequence extracted from entities.")
            return []

        metadata_keys = ["subject", "predicate", "object", "citation"]

        nodes = [
            TextNode(
                text=" ".join(knowledge[:3]),
                metadata={
                    "subject": knowledge[0],
                    "predicate": knowledge[1],
                    "object": knowledge[2],
                    "citation": knowledge[3].split("|") if knowledge[3] else [],
                },
                excluded_embed_metadata_keys=metadata_keys,
                excluded_llm_metadata_keys=metadata_keys,
            )
            for knowledge in knowledge_sequence
        ]

        faiss_index = faiss.IndexFlatL2(Settings.num_output)
        vector_store = FaissVectorStore(faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        retriever = index.as_retriever(similarity_top_k=self._similarity_top_k)
        nodes = retriever.retrieve(query_bundle)

        return nodes

    def _process_entities(
        self,
        query_str: str,
        handle_fn: Optional[Callable],
        handle_llm_prompt_template: Optional[BasePromptTemplate],
        cross_handle_policy: Optional[str] = "union",
        max_items: Optional[int] = 5,
        result_start_token: str = "KEYWORDS:",
    ) -> List[str]:
        """Get entities from query string."""
        # Skip if max_items is 0
        if max_items == 0:
            return []
        entities = super()._process_entities(
            query_str=query_str,
            handle_fn=handle_fn,
            handle_llm_prompt_template=handle_llm_prompt_template,
            cross_handle_policy=cross_handle_policy,
            max_items=max_items,
            result_start_token=result_start_token,
        )
        entities = self._clean_entities(entities)
        return entities

    def _clean_entities(self, entities):
        # clean entities by replacing non-alphanumeric characters with space and strip
        for entity in entities:
            cleaned_entity = re.sub("[^0-9a-zA-Z ]+", " ", entity).strip()
            if cleaned_entity != entity:
                entities.append(cleaned_entity)
        return entities

    async def _aprocess_entities(
        self,
        query_str: str,
        handle_fn: Optional[Callable],
        handle_llm_prompt_template: Optional[BasePromptTemplate],
        cross_handle_policy: Optional[str] = "union",
        max_items: Optional[int] = 5,
        result_start_token: str = "KEYWORDS:",
    ) -> List[str]:
        """Get entities from query string."""
        # Skip if max_items is 0
        if max_items == 0:
            return []
        entities = await super()._aprocess_entities(
            query_str=query_str,
            handle_fn=handle_fn,
            handle_llm_prompt_template=handle_llm_prompt_template,
            cross_handle_policy=cross_handle_policy,
            max_items=max_items,
            result_start_token=result_start_token,
        )
        entities = self._clean_entities(entities)
        return entities

    def _retrieve_keyword(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve in keyword mode."""
        if self._retriever_mode not in ["keyword", "keyword_embedding"]:
            return []
        # Get entities
        entities = self._get_entities(query_bundle.query_str)
        if self._verbose:
            print(f"> Entities extracted from query string: {entities}")
        # Before we enable embedding/semantic search, we need to make sure
        # we don't miss any entities that's synoynm of the entities we extracted
        # in string matching based retrieval in following steps, thus we expand
        # synonyms here.
        if len(entities) == 0:
            logger.info("> No entities extracted from query string.")
            return []

        # Get SubGraph from Graph Store as Knowledge Sequence
        knowledge_sequence, rel_map = self._get_knowledge_sequence(entities, query_bundle)

        return self._build_nodes(knowledge_sequence, rel_map, query_bundle)

    async def _aretrieve_keyword(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve in keyword mode."""
        if self._retriever_mode not in ["keyword", "keyword_embedding"]:
            return []
        # Get entities
        entities = await self._aget_entities(query_bundle.query_str)
        if self._verbose:
            print(f"> Entities extracted from query string: {entities}")
        # Before we enable embedding/semantic search, we need to make sure
        # we don't miss any entities that's synoynm of the entities we extracted
        # in string matching based retrieval in following steps, thus we expand
        # synonyms here.
        if len(entities) == 0:
            logger.info("> No entities extracted from query string.")
            return []

        # Get SubGraph from Graph Store as Knowledge Sequence
        knowledge_sequence, rel_map = await self._aget_knowledge_sequence(entities, query_bundle)

        return self._build_nodes(knowledge_sequence, rel_map, query_bundle)

    def _get_knowledge_sequence(
        self, entities: List[str], query_bundle: QueryBundle
    ) -> Tuple[List[List[str]], Optional[Dict[Any, Any]]]:
        """Get knowledge sequence from entities."""
        # Get SubGraph from Graph Store as Knowledge Sequence
        rel_map: Optional[Dict] = self._graph_store.get_rel_map(
            entities, self._graph_traversal_depth, limit=self._max_knowledge_sequence
        )
        logger.debug(f"rel_map: {rel_map}")

        # Build Knowledge Sequence
        memo = {}
        knowledge_sequence = []
        if rel_map:
            for rel_key, rel_values in rel_map.items():
                if rel_key in memo:
                    subj = memo[rel_key]
                else:
                    subj = self._get_best_rel_item(rel_key, query_bundle, entities)
                    memo[rel_key] = subj
                for rel, obj, citation in rel_values:
                    if rel in memo:
                        rel = memo[rel]
                    else:
                        rel = self._get_best_rel_item(rel, query_bundle)
                        memo[rel] = rel
                    if obj in memo:
                        obj = memo[obj]
                    else:
                        obj = self._get_best_rel_item(obj, query_bundle, entities)
                    knowledge_sequence.append((subj, rel, obj, citation))
        else:
            logger.info("> No knowledge sequence extracted from entities.")
            return [], None

        return knowledge_sequence, rel_map

    async def _aget_knowledge_sequence(
        self, entities: List[str], query_bundle: QueryBundle
    ) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        return self._get_knowledge_sequence(entities, query_bundle)

    def _get_best_rel_item(self, rel_items: str, query_bundle: QueryBundle, entities: List[str] | None = None) -> str:
        """Get best rel key."""
        rel_items = rel_items.split("|")
        rel_items = [rel_item for rel_item in rel_items if rel_item]
        # return rel_items[0]
        # if in entities
        rel_items_selected = []
        if entities:
            for entity in entities:
                if entity in rel_items:
                    rel_items_selected.append(entity)

        if rel_items_selected:
            rel_items = rel_items_selected

        if len(rel_items) == 1:
            return rel_items[0]

        nodes = [TextNode(text=rel_item) for rel_item in rel_items]

        faiss_index = faiss.IndexFlatL2(Settings.num_output)
        vector_store = FaissVectorStore(faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, callback_manager=CallbackManager())
        retriever = index.as_retriever(similarity_top_k=1)
        nodes = retriever.retrieve(query_bundle)
        return nodes[0].text
