import logging
import re
from typing import List
from uuid import uuid4
from metapub import PubMedFetcher

import pydot
from gard import GARD
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)
gard = GARD()


def onlineFullCitation(pmid:str, citation:str):
    '''
    Search PMC by PMID. Get article title, authors, 
    journal and abstract. (max 3 req per second w/out API, 10 with)
    Args:
        pmid(str): PMID to create citation for
        citation(str): original citation str
    Returns:
        citation(str): formatted citation
    '''
    full_citation = ''
    fetch = PubMedFetcher()
    try:
        article = fetch.article_by_pmid(pmid)
    except:
        return f"[{citation}](https://pubmed.ncbi.nlm.nih.gov/{pmid})"
    # extract metadata
    full_citation += article.title
    full_citation += f"\nJOURNAL: {article.journal}, {article.year}\nAUTHORS: {', '.join(article.authors)}\n"
    # add link
    full_citation += f"https://pubmed.ncbi.nlm.nih.gov/{pmid}\n"
    return full_citation


def format_citation(citation: str):
    '''
    Uses article ID to generate a URL for the source.
    Returns formatted link, with the citation as link text. 
    '''
    if citation.startswith("PMID:"):
        pmid = citation.removeprefix("PMID:")
        return onlineFullCitation(pmid, citation)
    elif citation.startswith("ORPHA:"):
        orpha_code = citation.removeprefix("ORPHA:")
        return f"[{citation}](https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=EN&Expert={orpha_code})"
    elif citation.startswith("OMIM:"):
        omim_identifier = citation.removeprefix("OMIM:")
        return f"[{citation}](https://www.omim.org/entry/{omim_identifier})"
    elif citation.startswith("UMLS:"):
        umls_identifier = citation.removeprefix("UMLS:")
        # temp fix for broken UMLS links
        return f"[{citation}](https://www.ncbi.nlm.nih.gov/medgen/?term={umls_identifier})"
        #return f"[{citation}](https://uts.nlm.nih.gov/metathesaurus.html#?searchString={umls_identifier})"
    elif citation.startswith("GARD:"):
        gard_identifier = citation.removeprefix("GARD:")
        return f"[{citation}]({gard.get_url(gard_identifier)})"
    else:
        return citation


def format_citations(citations: List[str]):
    citations_formatted = []
    for citation in citations:
        citations_formatted.append(format_citation(citation))
    return citations_formatted


async def get_formatted_sources(source_nodes:List[NodeWithScore]):
    '''
    Return formatted string of source numbers and corresponding urls.
    Return dict mapping original source numbers to new source numbers.
    '''
    references = "\n\n### Sources\n"
    citations_dict = {}
    sources_dict = {}

    for node in source_nodes:
        # get source number
        text = node.text
        source_number = int(text.split(":")[0].removeprefix("Source "))
        # get list of formatted source URLs
        citations = format_citations(node.metadata["citation"])
        # add URLs and source number to dict
        if len(citations) == 1:
            try:
                citations_dict[citations[0]].append(source_number)
            except KeyError:
                citations_dict[citations[0]] = [source_number]
        else:  # more than one citation for same source number :(
            for citation in citations:
                # add a unique source number for each citation
                try:
                    citations_dict[citation].append(source_number)
                except KeyError:
                    citations_dict[citation] = [source_number]
                source_number += 0.1  # make source number unique
        
    # consolidate citations w/ multiple source numbers to have just one
    #max = 0
    for c in citations_dict:
        # use min source number for citation
        new_source = min(citations_dict[c])
        references += f"[{new_source}] {str(citation)}"
        #max = new_source if new_source > max else max
        # if more than one source, track replaced numbers
        if len(citations_dict[c]) > 1:
            for source in citations_dict[c]:
                sources_dict[source] = new_source
        else:
            sources_dict[new_source] = new_source
    return references, sources_dict


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
