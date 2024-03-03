
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

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    response_message.content += response.get_formatted_sources()

    import plotly.graph_objects as go

    import networkx as nx

    G = nx.random_geometric_graph(200, 0.125)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
        title='<br>Network graph made with Python',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )

    elements = [cl.Plotly(name="chart", figure=fig, display="inline")]

    response_message.elements = elements

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
