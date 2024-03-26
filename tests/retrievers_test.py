import os

import pytest
from llama_index.core.prompts.base import PromptTemplate, PromptType
from llama_index.core.storage import StorageContext

from settings import configure_settings
from src.graph_stores import CustomNeo4jGraphStore
from src.retrievers import KG_RAG_KnowledgeGraphRAGRetriever

GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTIONS"))


@pytest.fixture
def retriever():
    configure_settings()
    graph_store = CustomNeo4jGraphStore(
        username="neo4j",
        password=os.environ["NEO4J_PASSWORD"],
        url="bolt://neo4j:7687",
        database="neo4j",
        node_label="S_PHENOTYPE",
        schema_cache_path="/data/rgd-chatbot/schema_cache.txt",
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
    @pytest.mark.skipif(GITHUB_ACTIONS, reason="This test won't run in Github Actions")
    def test_organizations(self, retriever):
        nodes = retriever.retrieve("What Canadian organizations can help with Duchenne Muscular Dystrophy?")
        texts = [node.text for node in nodes]
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization Stand for Duchenne Canada\nEmail: info@duchennecanada.org\nURL: https://duchennecanada.org/'
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization DuchenneXchange\nURL: https://www.duchennexchange.org/' in texts
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization Muscular Dystrophy Association\nAddress: \n222 S Riverside Plaza\nSuite 1500\nCity: Chicago\nCountry: United States\nEmail: resourcecenter@mdausa.org\nState: IL\nTollFree: 1-833-275-6321 (Helpline)\nURL: https://www.mda.org\nZipCode: 60606' in texts
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization Muscular Dystrophy Family Foundation\nAddress: \nP.O. Box 776\nCity: Carmel\nCountry: United States\nEmail: info@mdff.org\nPhone: +1-317-615-9140\nState: IN\nURL: https://mdff.org/\nZipCode: 46082' in texts
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization Muscular Dystrophy UK\nAddress: \n61A Great Suffolk Street\nCity: London\nCountry: United Kingdom\nEmail: info@musculardystrophyuk.org\nPhone:  (+44) 0 020 7803 4800\nTollFree: 0800 652 6352 (Helpline)\nURL: https://www.musculardystrophyuk.org/\nZipCode: SE1 0BU' in texts

    @pytest.mark.skipif(GITHUB_ACTIONS, reason="This test won't run in Github Actions")
    def test_incidence(self, retriever):
        nodes = retriever.retrieve("What is the incidence rate of Duchenne Muscular Dystrophy?")
        texts = [node.text for node in nodes]
        assert "DUCHENNE MUSCULAR DYSTROPHY has manifestation INCIDENCE OF 1 IN 3,500 BOYS" in texts

    @pytest.mark.skipif(GITHUB_ACTIONS, reason="This test won't run in Github Actions")
    def test_prevalence(self, retriever):
        nodes = retriever.retrieve("What is the prevalence of Duchenne Muscular Dystrophy in the UK?")
        texts = [node.text for node in nodes]
        assert "DUCHENNE MUSCULAR DYSTROPHY has prevalence PrevalenceClass: 1-9 / 100 000\nPrevalenceGeographic: United Kingdom\nPrevalenceQualification: Value and class\nPrevalenceValidationStatus: Validated\nValMoy: 4.14" in texts

    @pytest.mark.skipif(GITHUB_ACTIONS, reason="This test won't run in Github Actions")
    def test_genes(self, retriever):
        nodes = retriever.retrieve("What genes are associated with Duchenne Muscular Dystrophy?")
        texts = [node.text for node in nodes]
        assert "DUCHENNE MUSCULAR DYSTROPHY disease associated with gene DYSTROPHIN" in texts
