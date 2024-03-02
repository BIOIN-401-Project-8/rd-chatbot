
import logging
import sys

import chainlit as cl
from llama_index import StorageContext, load_index_from_storage
from llama_index.callbacks import CallbackManager
from llama_index.query_engine import CitationQueryEngine

from index import PERSIST_DIR
from service_context import get_service_context

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


@cl.on_chat_start
async def factory():
    callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    service_context = get_service_context(callback_manager=callback_manager)
    await cl.Message(content="Loaded service context").send()

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR / "vector_store_index")
    vector_store_index = load_index_from_storage(
        storage_context,
        service_context=service_context,
        show_progress=True,
    )
    await cl.Message(content="Loaded index from storage").send()

    query_engine = CitationQueryEngine.from_args(
        vector_store_index,
        similarity_top_k=10,
        citation_chunk_size=512,
        use_async=True,
        streaming=True,
    )
    cl.user_session.set("query_engine", query_engine)
    await cl.Message(content="Loaded query engine").send()


@cl.on_message
async def main(message: cl.Message):
    query_engine: CitationQueryEngine = cl.user_session.get("query_engine")
    response = await cl.make_async(query_engine.query)(message.content)
    response_message =  cl.Message(content="")

    import pickle
    with open("response.pkl", "wb") as f:
        pickle.dump(dir(response), f)

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    response_message.content += response.get_formatted_sources()

    await response_message.send()

    # for toke in response.source_nodes:
    #     response_message.content += f"{toke.node.get_text()}\n"
    # print("-- INTERNAL --")
    # print("Source nodes:")
    # for source_node in response.source_nodes:
    #     print(source_node.node.get_text())
    #     print(source_node.node.metadata)

# %%
# from pathlib import Path

# storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR / "graph_store_index")

# knowledge_graph_index = load_index_from_storage(
#     storage_context,
#     service_context=service_context,
#     show_progress=True,
# )
# from pyvis.network import Network

# g = knowledge_graph_index.get_networkx_graph()
# print(g)
# net = Network(notebook=True, cdn_resources="in_line", directed=True)
# net.from_nx(g)
# net.show("example_llm.html")
