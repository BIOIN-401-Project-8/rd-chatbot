from typing import Dict, List

from pyhpo import Ontology

Ontology()


def lookup_hpo_name(hpo_id: int | str):
    if isinstance(hpo_id, str):
        hpo_id = int(hpo_id.split(":")[-1])
    hpo_term = Ontology[hpo_id]
    return hpo_term.name


def textualize_phenotype(phenotype: dict):
    if not phenotype["m__N_Name"]:
        return None
    phenotype_description = []
    phenotype_description.append(phenotype["m__N_Name"])
    if phenotype["r_Frequency"]:
        frequency = lookup_hpo_name(phenotype["r_Frequency"])
        phenotype_description.append(f"Frequency: {frequency}")
    if phenotype["r_Onset"]:
        onset = lookup_hpo_name(phenotype["r_Onset"])
        phenotype_description.append(f"Onset: {onset}")
    return "\n".join(phenotype_description)


def extract_citations(text: str | list[str]):
    if not text:
        return []
    if isinstance(text, list):
        return text
    citations = text.split(",")
    citations[0] = citations[0].removeprefix("[")
    citations[-1] = citations[-1].removesuffix("]")
    return citations


def textualize_phenotypes(phenotypes: list[dict]):
    rel_map: Dict[str, List[List[str]]] = {}

    for phenotype in phenotypes:
        if phenotype["n__N_Name"] not in rel_map:
            rel_map[phenotype["n__N_Name"]] = []
        obj = textualize_phenotype(phenotype)
        if not obj:
            continue
        citations = extract_citations(phenotype["r_Reference"])
        rel_map[phenotype["n__N_Name"]].append(("has phenotype", obj, "|".join(citations)))
    return rel_map


def textualize_prevalence(prevalence: dict):
    if not prevalence["n_PrevalenceClass"]:
        return None
    prevalence_description = []
    prevalence_description.append(f"PrevalenceClass: {prevalence['n_PrevalenceClass']}")
    if prevalence["n_PrevalenceGeographic"]:
        prevalence_description.append(f"PrevalenceGeographic: {prevalence['n_PrevalenceGeographic']}")
    if prevalence["n_PrevalenceQualification"]:
        prevalence_description.append(f"PrevalenceQualification: {prevalence['n_PrevalenceQualification']}")
    if prevalence["n_PrevalenceValidationStatus"]:
        prevalence_description.append(f"PrevalenceValidationStatus: {prevalence['n_PrevalenceValidationStatus']}")
    if prevalence["n_ValMoy"]:
        prevalence_description.append(f"ValMoy: {prevalence['n_ValMoy']}")
    return "\n".join(prevalence_description)


def textualize_prevelances(prevalences: list[dict]):
    rel_map: Dict[str, List[List[str]]] = {}
    for prevalence in prevalences:
        if prevalence["m__N_Name"] not in rel_map:
            rel_map[prevalence["m__N_Name"]] = []
        citations = extract_citations(prevalence["n_Source"])
        rel_map[prevalence["m__N_Name"]].append(
            (
                "has prevalence",
                textualize_prevalence(prevalence),
                "|".join(citations),
            )
        )
    return rel_map


def textualize_organization(organization: dict):
    organization_description = [organization["Name"]]
    if organization["Address1"] or organization["Address2"]:
        organization_description.append(f"Address: ")
        organization_description.append(f"{organization['Address1']}" if organization["Address1"] else "")
        organization_description.append(f"{organization['Address2']}" if organization["Address2"] else "")
    if organization["City"]:
        organization_description.append(f"City: {organization['City']}")
    if organization["Country"]:
        organization_description.append(f"Country: {organization['Country']}")
    if organization["Email"]:
        organization_description.append(f"Email: {organization['Email']}")
    if organization["Fax"]:
        organization_description.append(f"Fax: {organization['Fax']}")
    if organization["Phone"]:
        organization_description.append(f"Phone: {organization['Phone']}")
    if organization["State"]:
        organization_description.append(f"State: {organization['State']}")
    if organization["TollFree"]:
        organization_description.append(f"TollFree: {organization['TollFree']}")
    if organization["URL"]:
        organization_description.append(f"URL: {organization['URL']}")
    if organization["ZipCode"]:
        organization_description.append(f"ZipCode: {organization['ZipCode']}")
    return organization_description
