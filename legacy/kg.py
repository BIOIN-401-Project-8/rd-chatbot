# %%
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index.graph_stores import SimpleGraphStore

from IPython.display import Markdown, display

from llama_index.llms import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# bge-m3 embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    embed_batch_size=64,  # 50 .. 80
)


# ollama
llm = Ollama(model="starling-lm", base_url="http://ollama:11434", request_timeout=30.0, temperature=0.0)

service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm, chunk_size=512)

documents = SimpleDirectoryReader("data").load_data()

from llama_index.storage.storage_context import StorageContext

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# %%
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
)

# %%
query_engine = index.as_query_engine(include_text=False, response_mode="tree_summarize")
response = query_engine.query(
    "Tell me more about Interleaf",
)
print(response)
# %%
query_engine = index.as_query_engine(include_text=True, response_mode="tree_summarize")
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)


# %%
new_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    service_context=service_context,
    include_embeddings=True,
)

# %%
# query using top 3 triplets plus keywords (duplicate triplets are removed)
query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf",
)

# %%
## create graph
from pyvis.network import Network

g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("example.html")

# %%
