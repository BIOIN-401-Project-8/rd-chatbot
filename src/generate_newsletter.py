'''
This module generates a newsletter based on the disease provided by the user.
Newsletters contain citations and short summaries of PubMed articles related
to the specified disease, that have been published within a 30 day window.
'''

from datetime import date, timedelta
#from llama_index.core.prompts import PromptTemplate


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
    disease = extractDisease(query)
    output += f"#Recent Articles On {disease}\n {start_date} - {end_date}\n\n"

    # GET RELEVANT PMIDs
    pmids = getPMIDs(disease, (start_date, end_date))

    # if no new articles in knowledge graph
    if len(pmids) == 0:
        output += "Sorry, no new articles were found."
    
    else:
        # display article links only
        if offline:
            for pmid in pmids:
                output += f"https://pubmed.ncbi.nlm.nih.gov/{pmid}\n"
        
        # get & display full citations  
        else:
            output += onlineFullCitations(pmids, disease)

    return output


def extractDisease(query:str)->str:
    '''
    Attempts to extract a disease from a user query.

    Args:
        query(str): user input
    Returns:
        disease(str): extracted disease name
    '''
    # LLM?
    # lookup in csv?
    disease = ''
    return disease


def getPMIDs(disease:str, date_range:tuple)->list:
    '''
    Get a list of PMIDs of recent disease articles
    from knowledge graph.
    
    Args:
        disease(str): disease name
        date_range(tuple): start and end date
    Returns:
        pmids(list): PMIDs of articles about disease, 
                     within date window
    '''
    pmids = []
    return pmids

def onlineFullCitations(pmids:list, disease:str):
    '''
    Search PMC for PMIDs. Get article title, authors, 
    journal and abstract.

    Args:
        pmids(list): list of PMIDs to create citations for
        disease(str): disease name
    Returns:
        citations(str): formatted citations
    '''
    citations = ''

    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    add = f"{disease}[mesh]+AND+2009[pdat]"

    for pmid in pmids:
        # search for article
        # extract title
        # extract authors
        # extract journal
        # extract abstract
        abstract = ''
        summary = summarize(abstract, disease)
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
    SUMMARIZE_TEMPLATE = (
        "A scientific paragraph about {disease} is provided below. "
        "Summarize the paragraph for a human teenager. "
        "Limit your response to 50 words. "
        "---------------------\n"
        "PARAGRAPH: {abstract}\n"
        "---------------------\n"
    )

    summary = ''
    return summary