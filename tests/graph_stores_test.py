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

    def test_get_rel_map_organization(self, graph_store: CustomNeo4jGraphStore):
        rel_map = graph_store.get_rel_map_organization(["CAT EYE SYNDROME"], limit=2)
        rels = rel_map["CAT EYE SYNDROME|CES|CHROMOSOME 22 PARTIAL TETRASOMY|INV DUP(22)(Q11)|SCHMID-FRACCARO SYNDROME"]
        assert rels[0] == (
            "has organization",
            "Chromosome 22 Central - US Office\nAddress: \n7108 Partinwood Drive\nCity: Fuquay-Varina\nCountry: United States\nEmail: usinfo@c22c.org\nPhone: 919-567-8167\nState: NC\nURL: http://www.c22c.org\nZipCode: 27526",
            "",
        )
        assert rels[1] == (
            "has organization",
            "Unique – Rare Chromosome Disorder Support Group\nAddress: \nG1, The Stables\nStation Road West\nCountry: United Kingdom\nEmail: info@rarechromo.org\nPhone: +44 (0)1883 723356\nURL: https://www.rarechromo.org/",
            "",
        )

    def test_get_rel_map_phenotype(self, graph_store: CustomNeo4jGraphStore):
        rel_map = graph_store.get_rel_map_phenotype(["GRACILE SYNDROME"], limit=2)
        rels = rel_map[
            "GRACILE SYNDROME|FELLMAN DISEASE|FELLMAN SYNDROME|FINNISH LACTIC ACIDOSIS WITH HEPATIC HEMOSIDEROSIS|FINNISH LETHAL NEONATAL METABOLIC SYNDROME|FLNMS|GROWTH DELAY-AMINOACIDURIA-CHOLESTASIS-IRON OVERLOAD-LACTIC ACIDOSIS-EARLY DEATH SYNDROME|GROWTH RESTRICTION-AMINOACIDURIA-CHOLESTASIS-IRON OVERLOAD-LACTIC ACIDOSIS-EARLY DEATH SYNDROME|GROWTH RETARDATION, AMINOACIDURIA, CHOLESTASIS, IRON OVERLOAD, LACTIC ACIDOSIS AND EARLY DEATH"
        ]
        assert rels[0] == (
            "has phenotype",
            "DEATH IN EARLY ADULTHOOD\nFrequency: Frequent",
            "ORPHA:53693",
        )
        assert rels[1] == (
            "has phenotype",
            "INCREASED SERUM PYRUVIC ACID|INCREASED SERUM PYRUVATE\nOnset: Neonatal onset",
            "PMID:12215968",
        )

    def test_get_rel_map_prevalence(self, graph_store: CustomNeo4jGraphStore):
        rel_map = graph_store.get_rel_map_prevalence(["GRACILE SYNDROME"], limit=2)
        rels = rel_map["GRACILE SYNDROME"]
        assert rels[0] == (
            "has prevalence",
            "PrevalenceClass: 1-9 / 100 000\nPrevalenceGeographic: Finland\nPrevalenceQualification: Value and class\nPrevalenceValidationStatus: Validated\nValMoy: 2.0",
            "PMID:22970607",
        )
        assert rels[1] == (
            "has prevalence",
            "PrevalenceClass: <1 / 1 000 000\nPrevalenceGeographic: Finland\nPrevalenceQualification: Class only\nPrevalenceValidationStatus: Not yet validated",
            "PMID:22970607",
        )

    def test_get_rel_map_rel(self, graph_store: CustomNeo4jGraphStore):
        rel_map = graph_store.get_rel_map(["GRACILE SYNDROME"], limit=1)
        rels = rel_map["GRACILE SYNDROME"]
        assert rels[0] == (
            "has allelic variant",
            "BCS1L|GENE:617|HGNC:1020|OMIM:603647\nInterpretation: Conflicting interpretations of pathogenicity",
            "PMID:17403714|UMLS:C1864002",
        )
        assert rels[1] == (
            "mapped to",
            "MITOCHONDRIAL METABOLISM DISEASE",
            "UMLS:C1864002",
        )

    def test_get_rel_map_pubtator3(self, graph_store: CustomNeo4jGraphStore):
        rel_map = graph_store.get_rel_map_pubtator3(["GRACILE Syndrome"], limit=1)
        rels = rel_map['Fellman disease|Fellman syndrome|GAD|GRACILE|GRACILE Syndrome|GRACILE syndrome|GRACILE-like|GRACILE-like condition|GRACILE-like disorder|GRACILE-like syndrome|Gracile axonal dystrophy|Gracile syndrome|and early death (GRACILE) syndrome|atrophy of gracile and cuneate nuclei|degeneration of the gracile nucleus and|gad|gracile|gracile axonal dystrophy|gracile fasciculi|gracile fasciculus|gracile syndrome|spinal gracile axonal dystrophy|tuberculum gracile']
        assert rels[0] == (
            "associate",
            "protein gene product 9.5",
            "PMID:14648596",
        )
