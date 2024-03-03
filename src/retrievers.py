import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from llama_index.indices.query.embedding_utils import (
    get_top_k_embeddings, get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings)
from llama_index.retrievers import KnowledgeGraphRAGRetriever
from llama_index.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)


class KG_RAG_KnowledgeGraphRAGRetriever(KnowledgeGraphRAGRetriever):
    def _build_nodes(
        self, knowledge_sequence: List[str], rel_map: Optional[Dict[Any, Any]] = None,
        query_bundle: QueryBundle = None
    ) -> List[NodeWithScore]:
        """Build nodes from knowledge sequence."""
        if len(knowledge_sequence) == 0:
            logger.info("> No knowledge sequence extracted from entities.")
            return []
        # _new_line_char = "\n"
        # context_string = (
        #     f"The following are knowledge sequence in max depth"
        #     f" {self._graph_traversal_depth} "
        #     f"in the form of directed graph like:\n"
        #     f"`subject -[predicate]->, object, <-[predicate_next_hop]-,"
        #     f" object_next_hop ...`"
        #     f" extracted based on key entities as subject:\n"
        #     f"{_new_line_char.join(knowledge_sequence)}"
        # )
        # if self._verbose:
        #     print_text(f"Graph RAG context:\n{context_string}\n", color="blue")

        rel_node_info = {
            "kg_rel_map": rel_map,
            "kg_rel_text": knowledge_sequence,
        }
        metadata_keys = ["kg_rel_map", "kg_rel_text"]
        if self._graph_schema != "":
            rel_node_info["kg_schema"] = {"schema": self._graph_schema}
            metadata_keys.append("kg_schema")

        service_context = self.get_service_context()
        embed_model = service_context.embed_model

        query_node = TextNode(
            text=query_bundle.query_str
        )
        query_node = embed_model([query_node])[0]
        query_embedding = np.array(query_node.embedding)

        nodes = [
            TextNode(
                text=knowledge,
                metadata=rel_node_info,
                excluded_embed_metadata_keys=metadata_keys,
                excluded_llm_metadata_keys=metadata_keys,
            )
            for knowledge in knowledge_sequence
        ]
        nodes = embed_model(nodes)

        nodes = [
            NodeWithScore(
                node=node,
                score=float(query_embedding @ np.array(node.embedding)),
            )
            for node in nodes
        ]

        # get top k
        k = 10
        nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        nodes = nodes[:k]

        return nodes

    def _retrieve_keyword(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve in keyword mode."""
        if self._retriever_mode not in ["keyword", "keyword_embedding"]:
            return []
        # Get entities
        entities = self._get_entities(query_bundle.query_str)
        # Before we enable embedding/semantic search, we need to make sure
        # we don't miss any entities that's synoynm of the entities we extracted
        # in string matching based retrieval in following steps, thus we expand
        # synonyms here.
        if len(entities) == 0:
            logger.info("> No entities extracted from query string.")
            return []

        # Get SubGraph from Graph Store as Knowledge Sequence
        knowledge_sequence, rel_map = self._get_knowledge_sequence(entities)

        return self._build_nodes(knowledge_sequence, rel_map, query_bundle)

    async def _aretrieve_keyword(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Retrieve in keyword mode."""
        if self._retriever_mode not in ["keyword", "keyword_embedding"]:
            return []
        # Get entities
        entities = await self._aget_entities(query_bundle.query_str)
        # Before we enable embedding/semantic search, we need to make sure
        # we don't miss any entities that's synoynm of the entities we extracted
        # in string matching based retrieval in following steps, thus we expand
        # synonyms here.
        if len(entities) == 0:
            logger.info("> No entities extracted from query string.")
            return []

        # Get SubGraph from Graph Store as Knowledge Sequence
        knowledge_sequence, rel_map = await self._aget_knowledge_sequence(entities)

        return self._build_nodes(knowledge_sequence, rel_map, query_bundle)

    def _get_knowledge_sequence(
        self, entities: List[str]
    ) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        """Get knowledge sequence from entities."""
        # Get SubGraph from Graph Store as Knowledge Sequence
        rel_map: Optional[Dict] = self._graph_store.get_rel_map(
            entities, self._graph_traversal_depth, limit=self._max_knowledge_sequence
        )
        logger.debug(f"rel_map: {rel_map}")

        # Build Knowledge Sequence
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend(
                [" ".join(rel_obj) for rel_objs in rel_map.values() for rel_obj in rel_objs]
            )
        else:
            logger.info("> No knowledge sequence extracted from entities.")
            return [], None

        return knowledge_sequence, rel_map

    async def _aget_knowledge_sequence(
        self, entities: List[str]
    ) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        """Get knowledge sequence from entities."""
        # Get SubGraph from Graph Store as Knowledge Sequence
        # TBD: async in graph store
        rel_map: Optional[Dict] = self._graph_store.get_rel_map(
            entities, self._graph_traversal_depth, limit=self._max_knowledge_sequence
        )
        logger.debug(f"rel_map from GraphStore:\n{rel_map}")

        # Build Knowledge Sequence
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend(
                [" ".join(rel_obj) for rel_objs in rel_map.values() for rel_obj in rel_objs]
            )
        else:
            logger.info("> No knowledge sequence extracted from entities.")
            return [], None

        return knowledge_sequence, rel_map
