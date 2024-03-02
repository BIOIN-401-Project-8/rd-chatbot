
# %%
import logging
import sys

from llama_index import StorageContext, load_index_from_storage
from llama_index.query_engine import CitationQueryEngine

from index import PERSIST_DIR
from service_context import get_service_context

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

service_context = get_service_context()
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR / "vector_store_index")

#%%
vector_store_index = load_index_from_storage(
    storage_context,
    service_context=service_context,
    show_progress=True,
)
query_engine = CitationQueryEngine.from_args(
    vector_store_index,
    similarity_top_k=20,
    citation_chunk_size=512,
)

# %%
response = query_engine.query("""What genes cause Duchenne Muscular Dystrophy?""")
print(response)
print("-- INTERNAL --")
print("Source nodes:")
for source_node in response.source_nodes:
    print(source_node.node.get_text())
    print(source_node.node.metadata)

# %%
from pathlib import Path
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR / "graph_store_index")

knowledge_graph_index = load_index_from_storage(
    storage_context,
    service_context=service_context,
    show_progress=True,
)
from pyvis.network import Network

g = knowledge_graph_index.get_networkx_graph()
print(g)
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("example_llm.html")

# %%
