import os

from src.textualize import (
    extract_citations,
    lookup_hpo_name,
    textualize_organization,
    textualize_phenotype,
    textualize_prevalence,
)


class TestTextualize:
    def test_lookup_hpo_name_int(self):
        name = lookup_hpo_name(40281)
        assert name == "Very frequent"

    def test_lookup_hpo_name_str(self):
        name = lookup_hpo_name("HP:0040281")
        assert name == "Very frequent"

    def test_texualize_organization_empty(self):
        organization = {
            "n_Name": "",
            "n_URL": "",
            "n_Email": "",
            "n_Phone": "",
            "n_Fax": "",
            "n_TollFree": "",
            "n_Address1": "",
            "n_Address2": "",
            "n_City": "",
            "n_State": "",
            "n_ZipCode": "",
            "n_Country": "",
        }
        description = textualize_organization(organization)
        assert description is None

    def test_texualize_organization_name_only(self):
        organization = {
            "n_Address1": "",
            "n_Address2": "",
            "n_City": "",
            "n_Country": "",
            "n_Email": "",
            "n_Fax": "",
            "n_Name": "Chromosome 22 Central - US Office",
            "n_Phone": "",
            "n_State": "",
            "n_TollFree": "",
            "n_URL": "",
            "n_ZipCode": "",
        }
        description = textualize_organization(organization)
        assert description == "Chromosome 22 Central - US Office"

    def test_texualize_organization(self):
        organization = {
            "n_Address1": "7108 Partinwood Drive",
            "n_Address2": "",
            "n_City": "Fuquay-Varina",
            "n_Country": "United States",
            "n_Email": "usinfo@c22c.org",
            "n_Fax": "",
            "n_Name": "Chromosome 22 Central - US Office",
            "n_Phone": "919-567-8167",
            "n_State": "NC",
            "n_TollFree": "",
            "n_URL": "http://www.c22c.org",
            "n_ZipCode": "27526  ",
        }
        description = textualize_organization(organization)
        assert description == "Chromosome 22 Central - US Office\nAddress: \n7108 Partinwood Drive\nCity: Fuquay-Varina\nCountry: United States\nEmail: usinfo@c22c.org\nPhone: 919-567-8167\nState: NC\nURL: http://www.c22c.org\nZipCode: 27526"

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
        assert description == "PrevalenceClass: 1-9 / 100 000"

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
        assert (
            description
            == "PrevalenceClass: 1-9 / 100 000\nPrevalenceGeographic: Finland\nPrevalenceQualification: Value and class\nPrevalenceValidationStatus: Validated\nValMoy: 2.0"
        )

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
