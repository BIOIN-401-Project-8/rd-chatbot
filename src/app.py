
import logging
import os
import sys


import chainlit as cl
import networkx as nx
import plotly.graph_objects as go
from llama_index import StorageContext
from llama_index.callbacks import CallbackManager
from llama_index.prompts import PromptTemplate
from llama_index.prompts.base import PromptType
from llama_index.query_engine import CitationQueryEngine

from graph_stores import CustomNeo4jGraphStore
from query_engine import CustomCitationQueryEngine
from retrievers import KG_RAG_KnowledgeGraphRAGRetriever
from service_context import get_service_context

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


@cl.on_chat_start
async def factory():
    callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
    service_context = get_service_context(callback_manager=callback_manager)

    graph_store = CustomNeo4jGraphStore(
        username="neo4j",
        password=os.environ["NEO4J_PASSWORD"],
        url="bolt://neo4j:7687",
        database="neo4j",
        node_label="Congenital and Genetic Diseases",
    )


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
        similarity_top_k=5,  # TODO: make this param actually do something, it's currently hardcoded somewhere
        citation_qa_template=CUSTOM_CITATION_QA_TEMPLATE,
        citation_refine_template=None,
        use_async=True,
        streaming=True,
        verbose=True,
    )
    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine: CitationQueryEngine = cl.user_session.get("query_engine")
    response = await cl.make_async(query_engine.query)(message.content)
    response_message =  cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    response_message.content += response.get_formatted_sources()

    response_message.content += "\n".join(
        [
            f"{node.score:.2f}: {node.text}"
            for node in sorted(response.source_nodes, key=lambda x: x.score, reverse=True)[:5]
        ]
    )

    add_graph(response_message)

    await response_message.send()


def add_graph(response_message):
    # TODO: experiment with unsafe_allow_html = false, and send a interactive neo4j graph / networkx graph
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
            x=0.005, y=-0.002 ),
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )

    elements = [cl.Plotly(name="chart", figure=fig, display="inline")]

    response_message.elements = elements
