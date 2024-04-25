import gzip
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List
from uuid import uuid4

import pydot
from gard import GARD
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import NodeWithScore
from pybtex.database import parse_string
from pybtex.plugin import find_plugin

logger = logging.getLogger(__name__)
gard = GARD()


def bib_to_apa7_html(bibtex):
    bibliography = parse_string(bibtex, "bibtex")
    formatted_bib = APA.format_bibliography(bibliography)
    return "\n".join(entry.text.render(TEXT) for entry in formatted_bib)


def find_text(element, tag):
    e = element.find(tag)
    if e is not None:
        return e.text
    return ""


APA = find_plugin("pybtex.style.formatting", "apa")()
TEXT = find_plugin("pybtex.backends", "text")()


def pmid_to_bib(pmid):
    pmid = int(pmid)
    padded_pmid = f"{pmid:08d}"
    with gzip.open(
        f"/data/Archive/pubmed/Archive/{padded_pmid[:2]}/{padded_pmid[2:4]}/{padded_pmid[4:6]}/{pmid}.xml.gz"
    ) as f:
        xml = f.read()
    root = ET.fromstring(xml)
    authors = []
    for a in root.findall(".//Author"):
        lastname = find_text(a, ".//LastName")
        forename = find_text(a, ".//ForeName")
        authors.append(f"{lastname}, {forename}")
    author = " and ".join(authors)
    if author == "":
        author = "Anonymous"
    title = find_text(root, ".//ArticleTitle")
    journal = find_text(root, ".//Journal/Title")
    year = find_text(root, ".//PubDate/Year")
    month = find_text(root, ".//PubDate/Month")
    doi = find_text(root, ".//ArticleId[@IdType='doi']")
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    bibtex = f"""
    @article{{{pmid},
            author = {{{author}}},
            title = {{{title}}},
            journal = {{{journal}}},
            year = {{{year}}},
            month = {{{month}}},
            doi = {{{doi}}},
            url = {{{url}}},
        }}
    """
    return bibtex


def generate_full_pmid_citation(pmid):
    bibtex = pmid_to_bib(pmid)
    citation = bib_to_apa7_html(bibtex)
    return citation


def format_citation(citation: str):
    if citation.startswith("PMID:"):
        pmid = citation.removeprefix("PMID:")
        try:
            return generate_full_pmid_citation(pmid)
        except FileNotFoundError:
            logger.exception(f"Could not find file for PMID {pmid}")
            return citation
    elif citation.startswith("ORPHA:"):
        orpha_code = citation.removeprefix("ORPHA:")
        return f"[{citation}](https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=EN&Expert={orpha_code})"
    elif citation.startswith("OMIM:"):
        omim_identifier = citation.removeprefix("OMIM:")
        return f"[{citation}](https://www.omim.org/entry/{omim_identifier})"
    elif citation.startswith("UMLS:"):
        umls_identifier = citation.removeprefix("UMLS:")
        return f"[{citation}](https://uts.nlm.nih.gov/metathesaurus.html#?searchString={umls_identifier})"
    elif citation.startswith("GARD:"):
        gard_identifier = citation.removeprefix("GARD:")
        return f"[{citation}]({gard.get_url(gard_identifier)})"
    else:
        return citation


def format_citations(citations: List[str]):
    citations_formatted = []
    for citation in citations:
        citations_formatted.append(format_citation(citation))
    return ", ".join(citations_formatted)


def generate_bibliography(source_nodes: List[NodeWithScore], source_order: List[int]):
    references = "\n\n### Sources\n"
    # deduplicate sources
    source_map = {}
    for node in source_nodes:
        source_number = int(node.text.split(":")[0].removeprefix("Source "))
        citations = node.metadata["citation"]
        source_map[source_number] = [format_citation(citation) for citation in citations]

    inline_citation_map = defaultdict(list)
    citation_bibliography_number = {}
    for i, (source_number, citations) in enumerate(sorted(source_map.items(), key=lambda x: source_order.index(x[0]))):
        for citation in citations:
            if citation not in citation_bibliography_number:
                citation_bibliography_number[citation] = len(citation_bibliography_number) + 1
            bibliography_number = citation_bibliography_number[citation]
            inline_citation_map[source_number].append(bibliography_number)
            references += f"[{bibliography_number}] {citation}\n"

    return references, inline_citation_map


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


def smart_inline_citation_format(numbers: list[int]):
    # calculate consecutive ranges
    # join those with -
    # join all with ,
    numbers.sort()
    ranges = []
    start = numbers[0]
    end = numbers[0]
    for number in numbers[1:]:
        if number == end + 1:
            end = number
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = number
            end = number
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    cites = ", ".join(ranges)
    return f"[{cites}]"


def merge_adjacent_citations(content: str):
    # test case
    # GNE Myopathy is a rare genetic disorder characterized by progressive muscle weakness and atrophy, primarily affecting skeletal muscles. The condition is caused by mutations in the GNE gene, which encodes an enzyme involved in the synthesis of sialic acid, a crucial component of cell membranes and various glycoproteins [1-10], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26].\n\nThe symptoms of GNE Myopathy typically manifest during early childhood or adolescence and include muscle weakness, muscle atrophy, and abnormal electrical activity in muscles as detected by electromyography (EMG) tests [2], [3], [4], [5], [6]. Over time, the disease can lead to significant disability and impaired mobility.\n\nThe GNE gene has multiple allelic variants associated with GNE Myopathy, which are responsible for the different forms of the disorder observed in affected individuals [17], [18], [19], [20]. The specific variant determines the severity and progression of the disease, as well as the age of onset and other clinical features.\n\nIn summary, GNE Myopathy is a genetic disorder characterized by muscle weakness, atrophy, and electrical abnormalities in muscles due to mutations in the GNE gene. The condition affects skeletal muscles and can lead to significant disability over time.
    content = re.sub(r"\], \[", ", ", content)
    citations = re.findall(r"(\[[\d,\- ]+\])", content)
    print(citations)
    mapping = {}
    for cite in citations:
        x = cite.removeprefix("[").removesuffix("]").split(",")
        numbers = set()
        for a in x:
            if "-" in a:
                start, end = map(int, a.split("-"))
                numbers.update(range(start, end + 1))
            else:
                numbers.add(int(a))
        numbers = list(numbers)
        print(numbers)
        mapping[cite] = smart_inline_citation_format(numbers)
    for x, y in mapping.items():
        content = content.replace(x, y)
    return content


def expand_citations(content: str):
    # (Sources 9-12) -> (Source 9, Source 10, Source 11, Source 12)
    sources = re.findall(r"Sources* ([\d]+)-([\d]+)", content)
    for start, end in sources:
        numbers = list(range(int(start), int(end) + 1))
        if f"Sources {start}-{end}" in content:
            content = content.replace(f"Sources {start}-{end}", ", ".join([f"Source {x}" for x in numbers]))
        else:
            content = content.replace(f"Source {start}-{end}", ", ".join([f"Source {x}" for x in numbers]))
    # (Sources 5, 6, 7) -> (Source 5), (Source 6), (Source 7)
    sources = re.findall(r"Sources ([\d, ]+)", content)
    for source in sources:
        numbers = [int(x) for x in source.split(",")]
        if len(numbers) == 1:
            content = content.replace(f"Sources {source}", f"Source {numbers[0]}")
        else:
            content = content.replace(f"Sources {source}", ", ".join([f"Source {x}" for x in numbers]))
    return content
