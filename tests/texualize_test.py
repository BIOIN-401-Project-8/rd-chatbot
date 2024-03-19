import os

from src.textualize import extract_phenotype_citations, lookup_hpo_name, textualize_phenotype

GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTIONS"))


class TestTextualize:
    def test_lookup_hpo_name_int(self):
        name = lookup_hpo_name(40281)
        assert name == "Very frequent"

    def test_lookup_hpo_name_str(self):
        name = lookup_hpo_name("HP:0040281")
        assert name == "Very frequent"

    def test_texualize_phenotype_empty(self):
        phenotype = {"m__N_Name": "", "r_Frequency": "", "r_Onset": ""}
        description = textualize_phenotype(phenotype)
        assert description is None

    def test_texualize_phenotype_name_only(self):
        phenotype = {
            "m__N_Name": "ELEVATED SERUM FERRITIN",
            "r_Frequency": "",
            "r_Onset": ""
        }
        description = textualize_phenotype(phenotype)
        assert description == "ELEVATED SERUM FERRITIN"

    def test_texualize_phenotype(self):
        phenotype = {
            "m__N_Name": "ELEVATED SERUM FERRITIN",
            "r_Frequency": "HP:0040281",
            "r_Onset": "HP:0003623",
        }
        description = textualize_phenotype(phenotype)
        assert description == "ELEVATED SERUM FERRITIN\nFrequency: Very frequent\nOnset: Neonatal onset"

    def test_extract_phenotype_citations_empty(self):
        phenotype = {"r_Reference": ""}
        citations = extract_phenotype_citations(phenotype)
        assert citations == []

    def test_extract_phenotype_citations_single(self):
        phenotype = {"r_Reference": "[PMID:12215968]"}
        citations = extract_phenotype_citations(phenotype)
        assert citations == ["PMID:12215968"]

    def test_extract_phenotype_citations_multiple(self):
        phenotype = {"r_Reference": "[PMID:12215968,ORPHA:53693]"}
        citations = extract_phenotype_citations(phenotype)
        assert citations == ["PMID:12215968", "ORPHA:53693"]
