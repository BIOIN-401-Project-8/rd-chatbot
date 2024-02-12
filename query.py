
# %%
import logging
import sys

from llama_index import StorageContext, load_index_from_storage
from llama_index.query_engine import CitationQueryEngine

from index import PERSIST_DIR
from src.service_context import get_service_context

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

service_context = get_service_context()
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(
    storage_context,
    service_context=service_context,
    show_progress=True,
)
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=10,
    citation_chunk_size=512,
)

# %%
response = query_engine.query("What genes are associated with Duchenne muscular dystrophy?")
print(response)
print("-- INTERNAL --")
print("Source nodes:")
for source_node in response.source_nodes:
    print(source_node.node.get_text())

# %%
