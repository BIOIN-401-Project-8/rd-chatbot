import logging
import re
from typing import List
from uuid import uuid4

import pydot
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


def format_citation(citation: str):
    if citation.startswith("PMID:"):
        pmid = citation.removeprefix("PMID:")
        return f"[{citation}](https://pubmed.ncbi.nlm.nih.gov/{pmid})"
    elif citation.startswith("ORPHA:"):
        orpha_code = citation.removeprefix("ORPHA:")
        return f"[{citation}](https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=EN&Expert={orpha_code})"
    elif citation.startswith("OMIM:"):
        omim_identifier = citation.removeprefix("OMIM:")
        return f"[{citation}](https://www.omim.org/entry/{omim_identifier})"
    elif citation.startswith("UMLS:"):
        umls_identifier = citation.removeprefix("UMLS:")
        return f"[{citation}](https://uts.nlm.nih.gov/metathesaurus.html#?searchString={umls_identifier})"
        # TODO: verify this link
    else:
        return citation


def format_citations(citations: List[str]):
    citations_formatted = []
    for citation in citations:
        citations_formatted.append(format_citation(citation))
    return ", ".join(citations_formatted)


def format_source(node: NodeWithScore):
    text = node.text
    source_number = int(text.split(":")[0].removeprefix("Source "))
    source = node.text.split(":")[1].split("\n")[0].strip()
    citation = format_citations(node.metadata["citation"])
    return f"[{source_number}] {citation} {source} ({node.score:.2f})"


async def get_formatted_sources(source_nodes: List[NodeWithScore]):
    references = "\n\n### Sources\n"
    references += "\n".join(
        [
            format_source(node)
            for node in source_nodes
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
