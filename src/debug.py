#%%
import logging
from typing import Dict, List, Optional, Tuple
from llama_index import StorageContext, load_index_from_storage
from llama_index.callbacks import CallbackManager
from llama_index.query_engine import CitationQueryEngine
from llama_index.retrievers import KnowledgeGraphRAGRetriever, VectorIndexRetriever
from llama_index.indices import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from typing import Any
import sys
import os
from index import PERSIST_DIR
from service_context import get_service_context
#%%
# use DEBUG level logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
#%%

service_context = get_service_context()


#%%
class CustomNeo4jGraphStore(Neo4jGraphStore):
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "neo4j",
        node_label: str = "Entity",
        **kwargs: Any,
    ) -> None:
        try:
            import neo4j
        except ImportError:
            raise ImportError("Please install neo4j: pip install neo4j")
        self.node_label = node_label
        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.schema = ""
        self.structured_schema: Dict[str, Any] = {}
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )
        # Set schema
        try:
            self.refresh_schema()
        except neo4j.exceptions.ClientError:
            raise ValueError(
                "Could not use APOC procedures. "
                "Please ensure the APOC plugin is installed in Neo4j and that "
                "'apoc.meta.data()' is allowed in Neo4j configuration "
            )
        # Create constraint for faster insert and retrieval
        try:  # Using Neo4j 5
            self.query(
            f"""
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{self.node_label}`) REQUIRE n.id IS UNIQUE;
                """
            )
        except Exception:  # Using Neo4j <5
            self.query(
                f"""
                CREATE CONSTRAINT IF NOT EXISTS ON (n:`{self.node_label}`) ASSERT n.id IS UNIQUE;
                """
            )

    def get_rel_map(self, subjs: List[str] | None = None, depth: int = 2, limit: int = 30) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: subj -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+

        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map

        subjs = [subj.upper() for subj in subjs]

        query = (
            f"""MATCH p=(n1:`{self.node_label}`)-[*1..{depth}]->() """
            f"""{"WHERE apoc.coll.intersection(apoc.convert.toList(n1.N_Name), $subjs)" if subjs else ""} """
            "UNWIND relationships(p) AS rel "
            "WITH n1._N_Name AS subj, p, apoc.coll.flatten(apoc.coll.toSet("
            "collect([type(rel), rel.name, endNode(rel)._N_Name, endNode(rel)._I_GENE]))) AS flattened_rels "
            f"RETURN subj, collect(flattened_rels) AS flattened_rels LIMIT {limit}"
        )

        data = list(self.query(query, {"subjs": subjs}))
        if not data:
            return rel_map

        for record in data:
            # replace _ with space
            flattened_rels = list(set(
                tuple([str(flattened_rel[1]) or str(flattened_rel[0]), str(flattened_rel[2]) or str(flattened_rel[3])])
                 for flattened_rel in record["flattened_rels"]))
            rel_map[record["subj"]] = flattened_rels
        return rel_map

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.
        """
        from pathlib import Path
        if Path("schema.txt").exists():
            with open("schema.txt", "r") as f:
                self.schema = f.read()
        else:
            super().refresh_schema()
            with open("schema.txt", "w") as f:
                f.write(self.schema)

graph_store = CustomNeo4jGraphStore(
    username="neo4j",
    password=os.environ["NEO4J_PASSWORD"],
    url="bolt://neo4j:7687",
    database="neo4j",
    node_label="Congenital and Genetic Diseases"
)


#%%
storage_context = StorageContext.from_defaults(
  graph_store=graph_store,
)


from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

from llama_index.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)

CUSTOM_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "diseases from the text. Focus on extracting the diseases that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide diseases in the following comma-separated format: 'KEYWORDS: <diseases>'\n"
)
from llama_index.utils import print_text, truncate_text
logger = logging.getLogger(__name__)
from llama_index.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)
import numpy as np

class KG_RAG_KnowledgeGraphRAGRetriever(KnowledgeGraphRAGRetriever):
    def _build_nodes(
        self, knowledge_sequence: List[str], rel_map: Optional[Dict[Any, Any]] = None,
        query_bundle: QueryBundle = None
    ) -> List[NodeWithScore]:
        """Build nodes from knowledge sequence."""
        if len(knowledge_sequence) == 0:
            logger.info("> No knowledge sequence extracted from entities.")
            return []
        _new_line_char = "\n"
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
        k = 1000
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
retriever = KG_RAG_KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
    service_context=service_context,
    graph_traversal_depth=1,
    max_entities=1,
    max_synonyms=0,
    entity_extract_template=PromptTemplate(
    CUSTOM_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
        prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
    )
)
# %%
# index = KnowledgeGraphIndex.from_documents(
#     [],
#     service_context=service_context,
# )

# %%
import json
nodes = retriever.retrieve("When do people with DUCHENNE MUSCULAR DYSTROPHY die?")
# %%
with open("nodes.json", "w") as f:
    json.dump([{
        "text": node.text,
        "score": node.score,
    } for node in nodes], f, indent=2)

# %%
# query_engine = index.as_query_engine(
#     include_text=True, response_mode="tree_summarize",
#     verbose=True,
# )

from typing import Any, List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.core.base_retriever import BaseRetriever
from llama_index.core.response.schema import RESPONSE_TYPE
from llama_index.indices.base import BaseGPTIndex
from llama_index.node_parser import SentenceSplitter, TextSplitter
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.prompts.base import BasePromptTemplate
from llama_index.prompts.mixin import PromptMixinType
from llama_index.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.service_context import ServiceContext
from llama_index.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.query_engine.citation_query_engine import DEFAULT_CITATION_CHUNK_OVERLAP, DEFAULT_CITATION_CHUNK_SIZE, CITATION_QA_TEMPLATE, CITATION_REFINE_TEMPLATE

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

CUSTOM_CITATION_QA_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

CUSTOM_CITATION_REFINE_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. "
    "We have provided an existing answer: {existing_answer}"
    "Below are several numbered sources of information. "
    "Use them to refine the existing answer. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "\nBegin refining!"
    "\n------\n"
    "{context_msg}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)


query_engine = CustomCitationQueryEngine.from_args(
    service_context,
    retriever=retriever,
    similarity_top_k=5,
    citation_qa_template=CUSTOM_CITATION_QA_TEMPLATE,
    citation_refine_template=None,
    use_async=True,
    streaming=True,
    verbose=True,
)
# from llama_index.query_engine import RetrieverQueryEngine


# query_engine = RetrieverQueryEngine.from_args(
#     retriever,
#     service_context=service_context,
#     verbose=True,
# )

# %%
response = query_engine.query("When do people with Duchenne Muscular Dystrophy die?")
print(response)
print()
print("\n".join([
    f"{node.score:.2f}: {node.text}"
    for node in
    sorted(
        response.source_nodes, key=lambda x: x.score, reverse=True
    )[:5]
]))

# # %%
# service_context.embed_model([TextNode(
#     text="What is the incidence of Duchenne's Muscular Dystrophy?"
# )])

# %%
