import os

import pytest
from llama_index import PromptTemplate, StorageContext
from llama_index.prompts.base import PromptType

from src.graph_stores import CustomNeo4jGraphStore
from src.retrievers import KG_RAG_KnowledgeGraphRAGRetriever
from src.service_context import get_service_context


@pytest.fixture
def retriever():
    service_context = get_service_context()
    graph_store = CustomNeo4jGraphStore(
        username="neo4j",
        password=os.environ["NEO4J_PASSWORD"],
        url="bolt://neo4j:7687",
        database="neo4j",
        node_label="S_PHENOTYPE",
    )

    storage_context = StorageContext.from_defaults(
        graph_store=graph_store,
    )

    CUSTOM_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
        "A question is provided below. Given the question, extract up to {max_keywords} "
        "diseases from the text. Focus on extracting the diseases that we can use "
        "to best lookup answers to the question. Avoid stopwords.\n"
        "---------------------\n"
        "QUESTION: {question}\n"
        "---------------------\n"
        "Provide diseases in the following comma-separated format: 'KEYWORDS: <diseases>'\n"
    )

    retriever = KG_RAG_KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        verbose=True,
        service_context=service_context,
        graph_traversal_depth=1,
        max_entities=2,
        max_synonyms=0,
        similarity_top_k=10,
        max_knowledge_sequence=1000,
        entity_extract_template=PromptTemplate(
            CUSTOM_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
            prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
        ),
    )
    return retriever


class TestKG_RAG_KnowledgeGraphRAGRetriever:
    @pytest.mark.skip(reason="This test is not yet implemented")
    def test_organizations(self, retriever):
        nodes = retriever.retrieve("What Canadian organizations can help with Duchenne Muscular Dystrophy?")
        assert len(nodes) > 0

    @pytest.mark.skip(reason="This test is not yet implemented")
    def test_incidence(self, retriever):
        nodes = retriever.retrieve("What is the incidence rate of Duchenne Muscular Dystrophy?")
        assert len(nodes) > 0

    @pytest.mark.skip(reason="This test is not yet implemented")
    def test_prevalence(self, retriever):
        nodes = retriever.retrieve("What is the prevalence of Duchenne Muscular Dystrophy in the UK?")
        assert len(nodes) > 0

    @pytest.mark.skip(reason="This test is not yet implemented")
    def test_genes(self, retriever):
        nodes = retriever.retrieve("What genes are associated with Duchenne Muscular Dystrophy?")
        assert len(nodes) > 0
