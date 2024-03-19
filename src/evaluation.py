# write a quick eval script to compare for a QA task just the llm vs with RAG
# %%
import cProfile
import logging
import os
import re
import sys
import time
from datetime import timedelta
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.storage import StorageContext

from callbacks import CustomLlamaIndexCallbackHandler
from chat_engine.citation_types import CitationChatMode
from citation import get_formatted_sources, get_source_graph, get_source_nodes
from graph_stores import CustomNeo4jGraphStore
from query_engine import CustomCitationQueryEngine
from retrievers import KG_RAG_KnowledgeGraphRAGRetriever
from settings import configure_settings
from translation import detect_language, translate

import pandas as pd

one_hop_true_false_v2_df = pd.read_csv(
    "/workspaces/rgd-chatbot/eval/data/KG_RAG/test_questions_one_hop_true_false_v2.csv"
)

# %%
one_hop_true_false_v2_df = one_hop_true_false_v2_df[["text", "label"]]
# %%
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from tqdm import tqdm
from settings import configure_settings

configure_settings()

llm = Settings.llm

prompt = PromptTemplate(
    "You are an expert biomedical researcher. Answer with True or False." "Question: {question}" "Answer: "
)

n = len(one_hop_true_false_v2_df)

for index, row in tqdm(one_hop_true_false_v2_df.head(n).iterrows(), total=n):
    start = time.time()
    response = llm.predict(prompt, question=row["text"])
    end = time.time()
    ans = bool("true" in response.lower() or "yes" in response.lower())
    one_hop_true_false_v2_df.loc[index, "llm_predict"] = response
    one_hop_true_false_v2_df.loc[index, "llm"] = ans
    one_hop_true_false_v2_df.loc[index, "llm_time"] = timedelta(seconds=end - start)

one_hop_true_false_v2_df
# %%
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
    verbose=False,
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
    verbose=False,
)
chat_engine = query_engine.as_chat_engine(
    chat_mode=CitationChatMode.CONDENSE_PLUS_CONTEXT,
    verbose=False,
)
from datetime import timedelta
import contextlib
import numpy as np
from llama_index.core.chat_engine.types import AgentChatResponse

one_hop_true_false_v2_df_shim = one_hop_true_false_v2_df[one_hop_true_false_v2_df["rag_predict"].isna()]

for index, row in tqdm(one_hop_true_false_v2_df_shim.iterrows(), total=len(one_hop_true_false_v2_df_shim)):
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
        start = time.time()
        try:
            response = chat_engine.chat("True or false " + row["text"])
        except Exception as e:
            response = str(e)
        else:
            response = response.response
        end = time.time()
        chat_engine.reset()
    ans = bool("true" in response.lower() or "yes" in response.lower())
    one_hop_true_false_v2_df_shim.loc[index, "rag_predict"] = response
    one_hop_true_false_v2_df_shim.loc[index, "rag_ans"] = ans
    one_hop_true_false_v2_df_shim.loc[index, "rag_time"] = timedelta(seconds=end - start)

#%%
one_hop_true_false_v2_df = pd.concat([one_hop_true_false_v2_df, one_hop_true_false_v2_df_shim])
# drop duplicates
one_hop_true_false_v2_df = one_hop_true_false_v2_df.drop_duplicates(subset=["text"], keep='last')
one_hop_true_false_v2_df
# %%
one_hop_true_false_v2_df.to_csv("/workspaces/rgd-chatbot/eval/results/KG_RAG/one_hop_true_false_v2_df.csv", index=False)

# %%
