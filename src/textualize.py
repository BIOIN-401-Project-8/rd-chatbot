import logging
from typing import Dict, List

from pyhpo import Ontology

Ontology()

logger = logging.getLogger(__name__)


def lookup_hpo_name(hpo_id: int | str):
    if isinstance(hpo_id, str):
        if hpo_id.startswith("HP:"):
            hpo_id = int(hpo_id.split(":")[-1])
        else:
            return hpo_id
    hpo_term = Ontology[hpo_id]
    return hpo_term.name


def lookup_hpo_names(hpo_ids: List[int | str] | int | str):
    if not isinstance(hpo_ids, list):
        hpo_ids = [hpo_ids]
    return [lookup_hpo_name(hpo_id) for hpo_id in hpo_ids]


def textualize_phenotype(phenotype: dict):
    if not phenotype["m__N_Name"]:
        return None
    phenotype_description = []
    phenotype_description.append(phenotype["m__N_Name"])
    if phenotype["r_Frequency"]:
        frequencies = lookup_hpo_names(phenotype["r_Frequency"])
        phenotype_description.append(f"Frequency: {','.join(frequencies)}")
    if phenotype["r_Onset"]:
        onsets = lookup_hpo_names(phenotype["r_Onset"])
        phenotype_description.append(f"Onset: {','.join(onsets)}")
    return "\n".join(phenotype_description)


def get_list(text: str | list[str]):
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
        phenotype_description = textualize_phenotype(phenotype)
        if not phenotype_description:
            continue
        citations = get_list(phenotype["r_Reference"])
        rel_map[phenotype["n__N_Name"]].append(("has phenotype", phenotype_description, "|".join(citations)))
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
        prevalence_description = textualize_prevalence(prevalence)
        if not prevalence_description:
            continue
        citations = get_list(prevalence["n_Source"])
        rel_map[prevalence["m__N_Name"]].append(
            (
                "has prevalence",
                prevalence_description,
                "|".join(citations),
            )
        )
    return rel_map


def textualize_organization(organization: dict):
    if not organization["n_Name"]:
        return None
    organization_description = [organization["n_Name"]]
    if organization["n_Address1"] or organization["n_Address2"]:
        organization_description.append(f"Address: ")
        if organization["n_Address1"]:
            organization_description.append(f"{organization['n_Address1']}")
        if organization["n_Address2"]:
            organization_description.append(f"{organization['n_Address2']}")
    if organization["n_City"]:
        organization_description.append(f"City: {organization['n_City']}")
    if organization["n_Country"]:
        organization_description.append(f"Country: {organization['n_Country']}")
    if organization["n_Email"]:
        organization_description.append(f"Email: {organization['n_Email']}")
    if organization["n_Fax"]:
        organization_description.append(f"Fax: {organization['n_Fax']}")
    if organization["n_Phone"]:
        organization_description.append(f"Phone: {organization['n_Phone']}")
    if organization["n_State"]:
        organization_description.append(f"State: {organization['n_State']}")
    if organization["n_TollFree"]:
        organization_description.append(f"TollFree: {organization['n_TollFree']}")
    if organization["n_URL"]:
        organization_description.append(f"URL: {organization['n_URL']}")
    if organization["n_ZipCode"]:
        organization_description.append(f"ZipCode: {organization['n_ZipCode'].strip()}")
    return "\n".join(organization_description)


def textualize_organizations(organizations: list[dict]):
    rel_map: Dict[str, List[List[str]]] = {}
    for organization in organizations:
        if organization["m__N_Name"] not in rel_map:
            rel_map[organization["m__N_Name"]] = []
        organization_description = textualize_organization(organization)
        if not organization_description:
            continue
        rel_map[organization["m__N_Name"]].append(("has organization", organization_description, ""))
    return rel_map


def textualize_rel(rel: dict):
    if not rel["m__N_Name"] and not rel["m__I_GENE"]:
        return None
    rel_description = []
    obj = ""
    if rel["m__N_Name"]:
        obj += rel["m__N_Name"]
    if rel["m__I_GENE"]:
        if obj:
            obj += "|"
        obj += rel["m__I_GENE"]
    rel_description.append(obj)
    if rel["r_interpretation"]:
        rel_description.append(f"Interpretation: {rel['r_interpretation']}")
    return "\n".join(rel_description)


def textualize_rels(rels: list[dict]):
    rel_map: Dict[str, List[List[str]]] = {}
    for rel in rels:
        subj = rel["n__N_Name"] or rel["n__I_GENE"]
        if subj not in rel_map:
            rel_map[subj] = []
        rel_description = textualize_rel(rel)
        if not rel_description:
            continue
        relationships = rel["r_name"]
        if isinstance(relationships, list):
            relationships = "|".join(relationships)
        relationships = relationships.replace("_", " ")
        citations = get_list(rel["r_citations"]) + get_list(rel["r_value"])
        rel_map[subj].append((relationships, rel_description, "|".join(citations)))
    return rel_map
