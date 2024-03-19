import os

from src.textualize import extract_citations, lookup_hpo_name, textualize_phenotype, textualize_prevalence


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
        phenotype = {"m__N_Name": "ELEVATED SERUM FERRITIN", "r_Frequency": "", "r_Onset": ""}
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

    def test_texualize_prevalence_empty(self):
        prevalence = {
            "n_PrevalenceClass": "",
            "n_PrevalenceGeographic": "",
            "n_PrevalenceQualification": "",
            "n_PrevalenceValidationStatus": "",
            "n_ValMoy": "",
        }
        description = textualize_prevalence(prevalence)
        assert description is None

    def test_texualize_prevalence_name_only(self):
        prevalence = {
            "n_PrevalenceClass": "1-9 / 100 000",
            "n_PrevalenceGeographic": "",
            "n_PrevalenceQualification": "",
            "n_PrevalenceValidationStatus": "",
            "n_ValMoy": "",
        }
        description = textualize_prevalence(prevalence)
        assert description == 'PrevalenceClass: 1-9 / 100 000'

    def test_texualize_prevalence(self):
        prevalence = {
            "m__N_Name": "GRACILE SYNDROME",
            "n_PrevalenceClass": "1-9 / 100 000",
            "n_PrevalenceGeographic": "Finland",
            "n_PrevalenceQualification": "Value and class",
            "n_PrevalenceValidationStatus": "Validated",
            "n_ValMoy": 2.0,
        }
        description = textualize_prevalence(prevalence)
        assert description == 'PrevalenceClass: 1-9 / 100 000\nPrevalenceGeographic: Finland\nPrevalenceQualification: Value and class\nPrevalenceValidationStatus: Validated\nValMoy: 2.0'

    def test_extract_citations_empty(self):
        citations = extract_citations("")
        assert citations == []

    def test_extract_citations_single(self):
        citations = extract_citations("[PMID:12215968]")
        assert citations == ["PMID:12215968"]

        citations = extract_citations("PMID:12215968")
        assert citations == ["PMID:12215968"]

        citations = extract_citations(["PMID:12215968"])
        assert citations == ["PMID:12215968"]

    def test_extract_citations_multiple(self):
        citations = extract_citations("[PMID:12215968,ORPHA:53693]")
        assert citations == ["PMID:12215968", "ORPHA:53693"]
