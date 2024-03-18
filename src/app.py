import cProfile
import logging
import os
import re
import sys
import time

import chainlit as cl
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.storage import StorageContext

from citation import get_formatted_sources, get_source_graph, get_source_nodes
from graph_stores import CustomNeo4jGraphStore
from query_engine import CustomCitationQueryEngine
from retrievers import KG_RAG_KnowledgeGraphRAGRetriever
from settings import configure_settings
from translation import detect_language, translate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


@cl.on_chat_start
async def factory():
    callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    configure_settings(callback_manager=callback_manager)

    graph_store = CustomNeo4jGraphStore(
        username="neo4j",
        password=os.environ["NEO4J_PASSWORD"],
        url="bolt://neo4j:7687",
        database="neo4j",
        node_label="S_PHENOTYPE",
        schema_cache_path="/data/rgd-chatbot/schema_cache.txt",
    )

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    CUSTOM_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
        "A question is provided below. Given the question, extract up to {max_keywords} "
        "diseases from the text. Focus on extracting the diseases that we can use "
        "to best lookup answers to the question. Avoid stopwords. Do not add an explanation.\n"
        "---------------------\n"
        "QUESTION: {question}\n"
        "---------------------\n"
        "Provide diseases in the following comma-separated format: 'KEYWORDS: disease1, disease2'\n"
    )

    retriever = KG_RAG_KnowledgeGraphRAGRetriever(
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
    chat_engine = query_engine.as_chat_engine(
        chat_mode=ChatMode.REACT,
        verbose=True,
    )
    cl.user_session.set("chat_engine", chat_engine)


def chat(chat_engine: BaseChatEngine, content: str, profile: bool = False):
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    response = chat_engine.chat(content)
    if profile:
        pr.disable()
        pr.dump_stats("profile.prof")
    return response


@cl.on_message
async def main(message: cl.Message):
    start = time.time()
    chat_engine: BaseChatEngine = cl.user_session.get("chat_engine")
    content = message.content

    detection = await detect_language(content)
    detected_language = detection["language"]
    if detected_language != "en" and detected_language is not None:
        content = await translate(content, target="en")

    response = await cl.make_async(chat)(chat_engine, content, profile=False)
    response_message = cl.Message(content="")

    content = response.response

    source_nodes = get_source_nodes(response, content)

    content = content.split("Sources:")[0].strip()
    content = re.sub(r"Source (\d+)", r"[\1]", content, flags=re.I)
    content = re.sub(r"\(\[", "[", content)
    content = re.sub(r"\]\)", "]", content)

    if detected_language != "en" and detected_language is not None:
        content = await translate(content, target=detected_language)

    if source_nodes:
        content += await get_formatted_sources(source_nodes)
        filename = get_source_graph(source_nodes)
        elements = [cl.Image(path=filename, display="inline", size="large")]
        response_message.elements = elements

    end = time.time()
    content += f"\n\n<small>{end - start:.2f} seconds</small>"
    response_message.content = content
    await response_message.send()
