'''
This module contains functions to generate a newsletter based on a disease provided by the user.
Newsletters contain citations and short summaries of PubMed articles related to the specified disease, 
that have been published within a 30 day window.
'''

from datetime import date, timedelta
from llama_index.core.prompts import PromptTemplate, PromptType
from metapub import PubMedFetcher
import re

from graph_stores import CustomNeo4jGraphStore
from pipelines import get_graph_store



def makeNewsletter(query:str, offline:bool):
    '''
    Main pipeline for newsletter creation.

    Args:
        query(str): disease name input by user
        offline(bool): if True, only include links to articles
                       else, use PMC API to get full citations
    Returns:
        output(str): formatted newsletter content
    '''
    output = ''

    # GET DATE RANGE
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    # EXTRACT DISEASE
    #diseases = extractDisease(query)
    disease = re.sub(r"[^A-Za-z0-9\s/\()\[\]\.-]+", '', query.lower())
    output += f"Recent Articles On {disease}\n{start_date} - {end_date}\n\n"

    # GET RELEVANT PMIDs
    #graph_store = get_graph_store()
    #pmids = getPMIDs(graph_store, disease, (start_date, end_date))
    pmids = ['33602943', '32717791', '25752877', '35165856']

    # if no new articles in knowledge graph
    if len(pmids) < 1:
        output += "Sorry, no new articles were found."
    
    else:
        # display article links only
        if offline:
            for pmid in pmids:
                output += f"[Article {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid})\n\n"
        
        # get & display full citations  
        else:
            output += onlineFullCitations(pmids, disease)
    return output


def extractDisease(query:str)->str:
    # waste of time maybe???
    # use LLM, csv file, or API?
    '''
    Attempts to extract a disease name and its
    synonyms from a user query.

    Args:
        query(str): user input
    Returns:
        disease(str): extracted disease names
    '''
    diseases = []
    # clean text, keep only ()[].-/
    query = re.sub(r"[^A-Za-z0-9\s/\()\[\]\.-]+", '', query.lower())
    # get disease synonyms from LLM
    SYNONYMS_TEMPLATE = PromptTemplate(
        "A disease name is provided below. "
        "Find synonyms for the disease name. "
        "---------------------\n"
        "DISEASE NAME: {query}\n"
        "---------------------\n"
        "Provide synonyms in the following comma-separated format: 'SYNONYMS: synonym1, synonym2'\n"
    )
    # query_words = '+'.join(query.split())
    # url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi?term={query_words}&db=pmc"
    return diseases


def getPMIDs(graph_store:CustomNeo4jGraphStore, disease:str, date_range:tuple)->list:
    '''
    Get a list of PMIDs of recent disease articles from knowledge graph.
    
    Args:
        graph_store(CustomNeo4jGraphStore): Neo4j knowledge graph to query
        disease(str): disease name
        date_range(tuple): start and end date
    Returns:
        pmids(list): PMIDs of articles about disease, 
                     within date window
    '''
    assert len(date_range) == 2 and date_range[0] < date_range[1], "Invalid date range."
    pmids = []
    # TODO: check format of nodes
    # DATE FORMAT??? n.date >= date('2024-03-01')
    cypher = f'''
    MATCH (n)
    WHERE n.name CONTAINS {disease} AND n.date >= {date_range[0]} AND n.date <= {date_range[1]}
    RETURN n
    '''
    nodes = graph_store.query(cypher)
    for n in nodes:
        # TODO: assert PMIDs are numeric & 8 digits
        pmids.append(n['pmid'])
    return pmids


def onlineFullCitations(pmids:list, disease:str):
    '''
    Search PMC by PMID. Get article title, authors, 
    journal and abstract. (max 3 req per second w/out API, 10 with)

    Args:
        pmids(list): list of PMIDs to create citations for
        disease(str): disease name
    Returns:
        citations(str): formatted citations
    '''
    # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}
    citations = ''
    fetch = PubMedFetcher()

    for pmid in pmids:
        try:
            article = fetch.article_by_pmid(pmid)
        except:
            continue
        # extract metadata
        citations += article.title + '\n' + '*' * 30
        citations += f"\nJOURNAL: {article.journal}, {article.year}\nAUTHORS: {', '.join(article.authors)}\n"
        # add link
        citations += f"https://pubmed.ncbi.nlm.nih.gov/{pmid}\n"
        # extract & summarize abstract
        summary = article.abstract[:25]
        #summary = summarize(article.abstract, disease)
        citations += summary + '\n\n'
    return citations


def summarize(abstract:str, disease:str)->str:
    '''
    Use LLM to convert an article abstract into an 
    easily readable 2 sentence summary.

    Args:
        abstract(str): article abstract to simplify
        disease(str): disease name
    Returns:
        summary(str): 2 sentence summary of abstract
    '''
    SUMMARIZE_TEMPLATE = PromptTemplate(
        "A scientific paragraph about {disease} is provided below. "
        "Summarize the paragraph for a 20 year old human. "
        "Include specific details like names and methods. "
        "Limit your response to 50 words. "
        "---------------------\n"
        "PARAGRAPH: {abstract}\n"
        "---------------------\n"
    )

    SUMMARIZE_REFINE_TEMPLATE = PromptTemplate(
        "Here is a short paragraph about {disease}: "
        "---------------------\n"
        "{summary}\n"
        "---------------------\n"
        "Use the information provided below to revise the paragraph. "
        "The revised paragraph should be readable by a 20 year old human. "
        "Do not increase the length of the paragraph. "
        "---------------------\n"
        "{context_msg}\n"
        "---------------------\n"
        "If the provided information is not informative, return the original paragraph."
    )

    summary = ''
    return summary