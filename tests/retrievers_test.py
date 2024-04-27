import os

import pytest
from conftest import GITHUB_ACTIONS

from pipelines import get_retriever_pipeline


@pytest.fixture
def retriever():
    return get_retriever_pipeline()


@pytest.mark.skipif(GITHUB_ACTIONS, reason="This test won't run in Github Actions")
class TestKG_RAG_KnowledgeGraphRAGRetriever:
    def test_organizations(self, retriever):
        nodes = retriever.retrieve("What Canadian organizations can help with Duchenne Muscular Dystrophy?")
        texts = [node.text for node in nodes]
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization Stand for Duchenne Canada\nEmail: info@duchennecanada.org\nURL: https://duchennecanada.org/'
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization DuchenneXchange\nURL: https://www.duchennexchange.org/' in texts
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization Muscular Dystrophy Association\nAddress: \n222 S Riverside Plaza\nSuite 1500\nCity: Chicago\nCountry: United States\nEmail: resourcecenter@mdausa.org\nState: IL\nTollFree: 1-833-275-6321 (Helpline)\nURL: https://www.mda.org\nZipCode: 60606' in texts
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization Muscular Dystrophy Family Foundation\nAddress: \nP.O. Box 776\nCity: Carmel\nCountry: United States\nEmail: info@mdff.org\nPhone: +1-317-615-9140\nState: IN\nURL: https://mdff.org/\nZipCode: 46082' in texts
        assert 'DUCHENNE MUSCULAR DYSTROPHY has organization Muscular Dystrophy UK\nAddress: \n61A Great Suffolk Street\nCity: London\nCountry: United Kingdom\nEmail: info@musculardystrophyuk.org\nPhone:  (+44) 0 020 7803 4800\nTollFree: 0800 652 6352 (Helpline)\nURL: https://www.musculardystrophyuk.org/\nZipCode: SE1 0BU' in texts

    def test_incidence(self, retriever):
        nodes = retriever.retrieve("What is the incidence rate of Duchenne Muscular Dystrophy?")
        texts = [node.text for node in nodes]
        assert "DUCHENNE MUSCULAR DYSTROPHY has manifestation INCIDENCE OF 1 IN 3,500 BOYS" in texts

    def test_prevalence(self, retriever):
        nodes = retriever.retrieve("What is the prevalence of Duchenne Muscular Dystrophy in the UK?")
        texts = [node.text for node in nodes]
        assert "DUCHENNE MUSCULAR DYSTROPHY has prevalence PrevalenceClass: 1-9 / 100 000\nPrevalenceGeographic: United Kingdom\nPrevalenceQualification: Value and class\nPrevalenceValidationStatus: Validated\nValMoy: 4.14" in texts

    def test_genes(self, retriever):
        nodes = retriever.retrieve("What genes are associated with Duchenne Muscular Dystrophy?")
        texts = [node.text for node in nodes]
        assert "DUCHENNE MUSCULAR DYSTROPHY disease associated with gene DYSTROPHIN" in texts

    def test_treat(self, retriever):
        nodes = retriever.retrieve("Do steroids treat Brachial plexus birth injury?")
        texts = [node.text for node in nodes]
        assert "Steroid- treat Brachial plexus birth injury" in texts

    def test_treats(self, retriever):
        nodes = retriever.retrieve("What treats PKU?")
        texts = [node.text for node in nodes]
        assert texts

    def test_GNE_myopathy(self, retriever):
        nodes = retriever.retrieve("My right leg hurts from GNE Myopathy, what do I do?")
        texts = [node.text for node in nodes]
        assert texts

    def test_extract_mentions(self, retriever):
        entities = retriever._get_entities("Is there a cure for cystic fibrosis?")
        assert "Cystic fibrosis" in entities
        entities = retriever._get_entities("My right leg hurts from GNE Myopathy, what do I do?")
        assert "GNE Myopathy" in entities

        # Only seems to work on Llama 70B
        # entities = retriever._get_entities("I get drunk without drinking alcohol, what rare disease do I have?")
        # assert "alcohol" in entities

        entities = retriever._get_entities("What treats PKU?")
        assert "PKU" in entities
        entities = retriever._get_entities("I have Maple Syrup Urine Disease. What organizations can help me?")
        assert "Maple Syrup Urine Disease" in entities
        entities = retriever._get_entities("Write a 250 word summary about Mucolipidosis IV.")
        assert "Mucolipidosis IV" in entities
        entities = retriever._get_entities("I have Klinefelter syndrome, what are the odds my children inherit it?")
        assert "Klinefelter syndrome" in entities
        entities = retriever._get_entities("How many people have Zellweger Spectrum Disorders?")
        assert "Zellweger Spectrum Disorders" in entities
        entities = retriever._get_entities("What causes L1 Syndrome?")
        assert "L1 Syndrome" in entities
        entities = retriever._get_entities("My child has GRACILE syndrome, how long will he live for?")
        assert "GRACILE syndrome" in entities

