# %%
import logging
import os
import sys

from llama_index import StorageContext
from llama_index.prompts import PromptTemplate
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

from graph_stores import CustomNeo4jGraphStore
from query_engine import CustomCitationQueryEngine
from retrievers import KG_RAG_KnowledgeGraphRAGRetriever
from service_context import get_service_context

# %%
# use DEBUG level logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# %%

service_context = get_service_context()


graph_store = CustomNeo4jGraphStore(
    username="neo4j",
    password=os.environ["NEO4J_PASSWORD"],
    url="bolt://neo4j:7687",
    database="neo4j",
    node_label="Congenital and Genetic Diseases",
)


# %%
storage_context = StorageContext.from_defaults(
    graph_store=graph_store,
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
    ),
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

response = query_engine.query("What is the incidence of Duchenne's Muscular Dystrophy?")
print(response)
print()
print(
    "\n".join(
        [
            f"{node.score:.2f}: {node.text}"
            for node in sorted(response.source_nodes, key=lambda x: x.score, reverse=True)[:5]
        ]
    )
)
