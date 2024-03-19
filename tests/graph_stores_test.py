import os

import pytest

from src.graph_stores import CustomNeo4jGraphStore

GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTIONS"))


@pytest.fixture
def graph_store():
    return CustomNeo4jGraphStore(
        username="neo4j",
        password=os.environ["NEO4J_PASSWORD"],
        url="bolt://neo4j:7687",
        database="neo4j",
        node_label="S_PHENOTYPE",
        schema_cache_path="/data/rgd-chatbot/schema_cache.txt",
    )


@pytest.mark.skipif(GITHUB_ACTIONS, reason="This test won't run in Github Actions")
class TestCustomNeo4jGraphStore:
    def test_get_rel_map_phenotype(self, graph_store: CustomNeo4jGraphStore):
        rel_map = graph_store.get_rel_map_phenotype(["GRACILE SYNDROME"], limit=2)
        rels = rel_map["GRACILE SYNDROME|FELLMAN DISEASE|FELLMAN SYNDROME|FINNISH LACTIC ACIDOSIS WITH HEPATIC HEMOSIDEROSIS|FINNISH LETHAL NEONATAL METABOLIC SYNDROME|FLNMS|GROWTH DELAY-AMINOACIDURIA-CHOLESTASIS-IRON OVERLOAD-LACTIC ACIDOSIS-EARLY DEATH SYNDROME|GROWTH RESTRICTION-AMINOACIDURIA-CHOLESTASIS-IRON OVERLOAD-LACTIC ACIDOSIS-EARLY DEATH SYNDROME|GROWTH RETARDATION, AMINOACIDURIA, CHOLESTASIS, IRON OVERLOAD, LACTIC ACIDOSIS AND EARLY DEATH"]
        assert rels[0] == (
            "has phenotype",
            "DEATH IN EARLY ADULTHOOD\nFrequency: Frequent",
            'ORPHA:53693',
        )
        assert rels[1] == (
            "has phenotype",
            "INCREASED SERUM PYRUVIC ACID|INCREASED SERUM PYRUVATE\nOnset: Neonatal onset",
            'PMID:12215968',
        )
