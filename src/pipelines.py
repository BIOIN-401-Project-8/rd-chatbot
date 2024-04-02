import os

import httpx
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.storage import StorageContext
from llama_index.llms.ollama import Ollama

from chat_engine.citation_types import CitationChatMode
from embeddings import SentenceTransformerEmbeddings
from graph_stores import CustomNeo4jGraphStore
from query_engine import CustomCitationQueryEngine
from retrievers import KG_RAG_KnowledgeGraphRAGRetriever


def get_graph_store():
    return CustomNeo4jGraphStore(
        username="neo4j",
        password=os.environ["NEO4J_PASSWORD"],
        url="bolt://neo4j:7687",
        database="neo4j",
        node_label="S_PHENOTYPE",
        schema_cache_path="/data/rgd-chatbot/schema_cache.txt",
    )


def get_retriever_pipeline(callback_manager: CallbackManager | None = None):
    Settings.llm = get_llm()
    Settings.embed_model, Settings.num_output = get_embed_model()
    Settings.callback_manager = callback_manager

    graph_store = get_graph_store()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    return get_retriever(storage_context)


def get_pipeline(callback_manager: CallbackManager | None = None):
    retriever = get_retriever_pipeline(callback_manager)

    query_engine = get_query_engine(retriever)
    chat_engine = query_engine.as_chat_engine(
        chat_mode=CitationChatMode.CONDENSE_PLUS_CONTEXT,
        verbose=True,
    )
    return chat_engine


def get_embed_model(embed_model_name: str = "intfloat/e5-base-v2"):
    return (
        SentenceTransformerEmbeddings(
            model_name_or_path=embed_model_name,
            embed_batch_size=16,
        ),
        768,
    )


def get_llm(llm_model_name: str = "starling-lm"):
    # Pulling the model with Ollama
    # TODO: display this as a progress bar
    httpx.post("http://ollama:11434/api/pull", json={"name": llm_model_name}, timeout=600.0)
    return Ollama(
        model=llm_model_name,
        base_url="http://ollama:11434",
        request_timeout=30.0,
        temperature=0.0,
    )


def get_query_engine(retriever: BaseRetriever):
    CUSTOM_CITATION_QA_TEMPLATE = PromptTemplate(
        "Please provide an answer based solely on the provided sources. "
        "When referencing information from a source, "
        "cite the appropriate source(s) using their corresponding numbers. "
        "Every answer should include at least one source citation. "
        "Only cite a source when you are explicitly referencing it. "
        "Do not list the sources again at the end. "
        "If none of the sources are helpful, you should indicate that. "
        "For example:\n"
        "Source 1:\n"
        "The sky is red in the evening and blue in the morning.\n"
        "Source 2:\n"
        "Water is wet when the sky is red.\n"
        "Query: When is water wet?\n"
        "Answer: Water will be wet when the sky is red (SOURCE 2), "
        "which occurs in the evening (SOURCE 1).\n"
        "DONE\n"
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
        "Answer: Water will be wet when the sky is red (SOURCE 2), "
        "which occurs in the evening (SOURCE 1).\n"
        "Now it's your turn. "
        "We have provided an existing answer: {existing_answer}"
        "Below are several numbered sources of information. "
        "Use them to refine the existing answer. "
        "If the provided sources are not helpful, you will repeat the existing answer."
        "Do not list the sources again at the end. "
        "\nBegin refining!"
        "\n------\n"
        "{context_msg}"
        "\n------\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    query_engine = CustomCitationQueryEngine.from_args(
        retriever=retriever,
        citation_qa_template=CUSTOM_CITATION_QA_TEMPLATE,
        citation_refine_template=CUSTOM_CITATION_REFINE_TEMPLATE,
        use_async=True,
        streaming=True,
        verbose=True,
    )

    return query_engine


def get_retriever(
    storage_context: StorageContext,
):
    CUSTOM_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
        "A question is provided below. Given the question, extract up to {max_keywords} "
        "diseases from the text. Focus on extracting the diseases that we can use "
        "to best lookup answers to the question. Avoid stopwords. Do not add an explanation.\n"
        "---------------------\n"
        "QUESTION: {question}\n"
        "---------------------\n"
        "Provide diseases in the following comma-separated format: 'KEYWORDS: disease1, disease2'\n"
    )
    return KG_RAG_KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        verbose=True,
        graph_traversal_depth=1,
        max_entities=2,
        max_synonyms=1,
        similarity_top_k=30,
        max_knowledge_sequence=1000,
        entity_extract_template=PromptTemplate(
            CUSTOM_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
            prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
        ),
    )
