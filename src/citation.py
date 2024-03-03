import re
from typing import List
from uuid import uuid4

import networkx as nx
import plotly.graph_objects as go
import pydot
from llama_index.core.response.schema import RESPONSE_TYPE
from llama_index.core.schema import NodeWithScore


def format_source(node: NodeWithScore):
    text = node.text
    source_number = int(text.split(":")[0].removeprefix("Source "))
    source = node.text.split(":")[1].strip()
    return f"[{source_number}] {source} ({node.score:.2f})"


async def get_formatted_sources(source_nodes: List[NodeWithScore]):
    references = "\n\n### Sources\n"
    references += "\n".join(
        [
            format_source(node)
            for node in sorted(
                source_nodes,
                key=lambda x: x.score,
                reverse=True,
            )
        ]
    )
    return references


def get_source_nodes(response: RESPONSE_TYPE, content: str):
    sources = get_sources(content)
    source_nodes = []
    for source_node in response.source_nodes:
        source = int(source_node.text.split(":")[0].removeprefix("Source "))
        if source in sources:
            source_nodes.append(source_node)
    return source_nodes


def get_sources(content: str):
    sources = set(map(int, re.findall(r"SOURCE (\d+)", content, re.I)))
    matches = re.findall(r"SOURCES ([\d, ]+)", content)
    if matches:
        sources.update(map(int, matches[0].split(",")))
    return sources


def get_source_graph(source_nodes: List[NodeWithScore]):
    graph = pydot.Dot("source_graph", graph_type="digraph")

    for node in source_nodes:
        subj = node.metadata["subject"]
        obj = node.metadata["object"]
        predicate = node.metadata["predicate"]
        graph.add_edge(pydot.Edge(subj, obj, label=predicate))

    filename = f".files/{uuid4()}.png"
    graph.write_png(filename)
    return filename
