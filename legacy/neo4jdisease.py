# %%
from llama_index import VectorStoreIndex, download_loader, SummaryIndex, KnowledgeGraphIndex
from service_context import get_service_context
#%%
GraphDBCypherReader = download_loader("GraphDBCypherReader")
service_context = get_service_context()
reader = GraphDBCypherReader(uri="neo4j+s://disease.ncats.io:7687", username="", password="", database="neo4j")

# %%
# query = """
#     MATCH (n)-[r]->(m)
#     RETURN n,r,m
#     LIMIT 1000
# """
query = """
    MATCH (n)
    RETURN n
    LIMIT 100000
"""

documents = reader.load_data(
    query,
    parameters={},
)
#%%
index = KnowledgeGraphIndex.from_documents(documents, service_context=service_context)
#%%
print(len(documents))
print(documents[101].text)

# %%
from llama_index.query_engine import CitationQueryEngine
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=10,
    citation_chunk_size=512,
)

response = query_engine.query(
    """
What is the GARD ID for the disease with the name Duchenne muscular dystrophy?
"""
)
print(response)
print("-- INTERNAL --")
print("Source nodes:")
for source_node in response.source_nodes:
    print(source_node.node.get_text())
    print(source_node.node.metadata)

# %%
