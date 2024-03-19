from itertools import chain
from typing import Any, Dict, List

from llama_index.graph_stores.neo4j import Neo4jGraphStore

from textualize import textualize_organization, textualize_prevalence, textualize_phenotypes


class CustomNeo4jGraphStore(Neo4jGraphStore):
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "neo4j",
        node_label: str = "Entity",
        schema_cache_path: str = "schema_cache.txt",
        **kwargs: Any,
    ) -> None:
        try:
            import neo4j
        except ImportError:
            raise ImportError("Please install neo4j: pip install neo4j")
        self.node_label = node_label
        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.schema = ""
        self.structured_schema: Dict[str, Any] = {}
        self.schema_cache_path = schema_cache_path
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError("Could not connect to Neo4j database. " "Please ensure that the url is correct")
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. " "Please ensure that the username and password are correct"
            )
        # Set schema
        try:
            self.refresh_schema()
        except neo4j.exceptions.ClientError:
            raise ValueError(
                "Could not use APOC procedures. "
                "Please ensure the APOC plugin is installed in Neo4j and that "
                "'apoc.meta.data()' is allowed in Neo4j configuration "
            )
        # Create constraint for faster insert and retrieval
        try:  # Using Neo4j 5
            self.query(
                f"""
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{self.node_label}`) REQUIRE n.id IS UNIQUE;
                """
            )
        except Exception:  # Using Neo4j <5
            self.query(
                f"""
                CREATE CONSTRAINT IF NOT EXISTS ON (n:`{self.node_label}`) ASSERT n.id IS UNIQUE;
                """
            )

    def get_rel_map(
        self, subjs: List[str] | None = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: subj -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+

        rel_map: Dict[str, List[List[str]]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map

        subjs = [subj.upper() for subj in subjs]

        query = (
            f"""MATCH p=(n1:`{self.node_label}`)-[*1..{depth}]->() """
            f"""{"WHERE apoc.coll.intersection(apoc.convert.toList(n1.N_Name), $subjs)" if subjs else ""} """
            "UNWIND relationships(p) AS rel "
            "WITH n1._N_Name AS subj, p, apoc.coll.flatten(apoc.coll.toSet("
            "collect([type(rel), rel.name, endNode(rel)._N_Name, endNode(rel)._I_GENE, rel.value, rel.citations, rel.Reference]))) AS flattened_rels "
            f"RETURN subj, collect(flattened_rels) AS flattened_rels LIMIT {limit}"
        )

        data = list(self.query(query, {"subjs": subjs}))
        if not data:
            return rel_map

        # for record in data:
        #     flattened_rels = list(
        #         set(
        #             tuple(
        #                 [
        #                     (str(flattened_rel[1]) or str(flattened_rel[0])).replace("_", " "),
        #                     str(flattened_rel[2]) or str(flattened_rel[3]),
        #                 ]
        #             )
        #             for flattened_rel in record["flattened_rels"]
        #         )
        #     )
        #     for subj in record["subj"].split("|"):
        #         if subj not in rel_map:
        #             rel_map[subj] = []
        #         rel_map[subj] += flattened_rels

        rel_map = self.get_rel_map_phenotype(subjs, limit)
        # rel_map_organization = self.get_rel_map_organization(subjs, depth, limit)
        # rel_map_prevalence = self.get_rel_map_prevalence(subjs, depth, limit)
        # for subj, rels in chain(rel_map_organization.items(), rel_map_prevalence.items()):
        #     if subj in rel_map:
        #         rel_map[subj] += rels
        #     else:
        #         rel_map[subj] = rels
        # return rel_map
        return rel_map

    # def get_rel_map_organization(
    #     self, subjs: List[str] | None = None, depth: int = 2, limit: int = 30
    # ) -> Dict[str, List[List[str]]]:
    #     rel_map: Dict[Any, List[Any]] = {}
    #     if subjs is None or len(subjs) == 0:
    #         return rel_map

    #     subjs = [subj.upper() for subj in subjs]

    #     query = f"""
    #         MATCH p=(n1:`{self.node_label}`)<-[:ORGANIZATION*1..{depth}]-(n2)
    #         {"WHERE apoc.coll.intersection(apoc.convert.toList(n1.N_Name), $subjs)" if subjs else ""}
    #         RETURN n1._N_Name AS _N_Name, n2.Address1 AS Address1, n2.Address2 AS Address2, n2.City AS City, n2.Country AS Country, n2.Email AS Email, n2.Fax as Fax, n2.Name as Name, n2.Phone as Phone, n2.State as State, n2.TollFree as TollFree, n2.URL as URL, n2.ZipCode as ZipCode
    #         LIMIT {limit}
    #     """
    #     organizations = list(self.query(query, {"subjs": subjs}))

    #     if not organizations:
    #         return rel_map

    #     for organization in organizations:
    #         organization_description = textualize_organization(organization)
    #         for obj in organization["_N_Name"].split("|"):
    #             if obj not in rel_map:
    #                 rel_map[obj] = []
    #             rel_map[obj].append(
    #                 (
    #                     "has organization",
    #                     "\n".join(organization_description),
    #                 )
    #             )
    #     return rel_map

    def get_rel_map_phenotype(
        self, subjs: List[str] | None = None, limit: int = 30
    ):
        if subjs is None or len(subjs) == 0:
            return {}

        subjs = [subj.upper() for subj in subjs]

        # I_CODE	[GARD:0000001,OMIM:603358,ORPHA:53693,ORPHANET:53693,UMLS:C1864002]
        # N_Name	[GRACILE SYNDROME,FELLMAN DISEASE,FELLMAN SYNDROME,FINNISH LACTIC ACIDOSIS WITH HEPATIC HEMOSIDEROSIS,FINNISH LETHAL NEONATAL METABOLIC SYNDROME,FLNMS,… Show all]
        # R_equivalentClass	http://purl.obolibrary.org/obo/MONDO_0011308
        # R_hasPhenotype	[HP:0003355,HP:0001511,HP:0004925,HP:0001396,HP:0003281,HP:0003452,HP:0003542,HP:0001319,HP:0000365,HP:0001394,HP:0001397,HP:0001994,HP:0003128,HP:0012… Show all]
        # R_rel	[UMLS:C1864002,UMLS:C0001125,UMLS:C0008370,UMLS:C0235988,UMLS:C0238621,UMLS:C0441748,UMLS:C1272097,UMLS:C1837902,UMLS:C2749200,UMLS:C3551739,UMLS:C5232… Show all]
        # _I_CODE	GARD:0000001|OMIM:603358|ORPHA:53693|ORPHANET:53693|UMLS:C1864002
        # _N_Name	GRACILE SYNDROME|FELLMAN DISEASE|FELLMAN SYNDROME|FINNISH LACTIC ACIDOSIS WITH HEPATIC HEMOSIDEROSIS|FINNISH LETHAL NEONATAL METABOLIC SYNDROME|FLNMS|… Show all

        # Aspect	P
        # Biocuration	[HPO:probinson[2013-02-18],ORPHA:orphadata[2020-06-08]]
        # DatabaseID	[OMIM:603358,ORPHA:53693]
        # DiseaseName	[#603358 GRACILE SYNDROME;;GROWTH RETARDATION, AMINO ACIDURIA, CHOLESTASIS, IRON OVERLOAD, LACTICACIDOSIS, AND EARLY DEATH;;FINNISH LETHAL NEONATAL MET… Show all]
        # Evidence	[PCS,TAS]
        # Frequency	HP:0040281
        # HPO_ID	HP:0003281
        # Onset	HP:0003623
        # Reference	[PMID:12215968,ORPHA:53693]
        # _neo4j_sync_from_id	14652003
        # created	1607626678882
        # source	584572e46
        # value	HP:0003281

        # I_CODE	[UMLS:C0241013,UMLS:C0743912,UMLS:C3854388,HP:0003281]
        # N_Name	[ELEVATED SERUM FERRITIN,HIGH FERRITIN LEVEL,HYPERFERRITINAEMIA,HYPERFERRITINEMIA,INCREASED FERRITIN,INCREASED PLASMA FERRITIN,INCREASED SERUM FERRITIN… Show all]
        # R_equivalentClass	f05b0ae0-3977-4491-ad14-c00e2bdf59d2
        # R_hasPhenotype	[HP:0003281]
        # R_rel	[UMLS:C1863727,UMLS:C4551514,UMLS:C0268647,UMLS:C0878682,UMLS:C1853733,UMLS:C1858664,UMLS:C1865614,UMLS:C1865616,UMLS:C3469186,UMLS:C4015067,UMLS:C4225… Show all]
        # R_subClassOf	[http://purl.obolibrary.org/obo/HP_0040133,UMLS:C0241013]
        # _I_CODE	UMLS:C0241013|UMLS:C0743912|UMLS:C3854388|HP:0003281
        # _N_Name	ELEVATED SERUM FERRITIN|HIGH FERRITIN LEVEL|HYPERFERRITINAEMIA|HYPERFERRITINEMIA|INCREASED FERRITIN|INCREASED PLASMA FERRITIN|INCREASED SERUM FERRITIN… Show all

        query = f"""
            MATCH p=(n:`{self.node_label}`)-[r:R_hasPhenotype]->(m)
            {"WHERE apoc.coll.intersection(apoc.convert.toList(n.N_Name), $subjs)" if subjs else ""}
            RETURN n._N_Name AS n__N_Name, m._N_Name AS m__N_Name, r.Frequency AS r_Frequency, r.Onset AS r_Onset, r.Reference AS r_Reference
            LIMIT {limit}
        """
        phenotypes = list(self.query(query, {"subjs": subjs}))

        if not phenotypes:
            return {}

        return textualize_phenotypes(phenotypes)

    # def get_rel_map_prevalence(
    #     self, subjs: List[str] | None = None, depth: int = 2, limit: int = 30
    # ) -> Dict[str, List[List[str]]]:
    #     rel_map: Dict[Any, List[Any]] = {}
    #     if subjs is None or len(subjs) == 0:
    #         return rel_map

    #     subjs = [subj.upper() for subj in subjs]

    #     query = f"""
    #         MATCH p=(n1:`{self.node_label}`)<-[:PREVALENCE*1..{depth}]-(n2)
    #         {"WHERE apoc.coll.intersection(apoc.convert.toList(n1.N_Name), $subjs)" if subjs else ""}
    #         RETURN n1._N_Name AS _N_Name, n2.PrevalenceClass AS PrevalenceClass, n2.PrevalenceGeographic AS PrevalenceGeographic, n2.PrevalenceQualification AS PrevalenceQualification, n2.PrevalenceValidationStatus AS PrevalenceValidationStatus, n2.Source AS Source, n2.ValMoy as ValMoy
    #         LIMIT {limit}
    #     """
    #     prevalences = list(self.query(query, {"subjs": subjs}))

    #     if not prevalences:
    #         return rel_map

    #     for prevalence in prevalences:
    #         prevalence_description = textualize_prevalence(prevalence)
    #         # TODO: unflip the relation
    #         # TODO: first do vector similarity on subjects then, keep the most relevant subject only
    #         for obj in prevalence["_N_Name"].split("|"):
    #             if obj not in rel_map:
    #                 rel_map[obj] = []
    #             rel_map[obj].append(
    #                 (
    #                     "has prevalence",
    #                     "\n".join(prevalence_description),
    #                 )
    #             )
    #     return rel_map

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.
        """
        from pathlib import Path

        if Path(self.schema_cache_path).exists():
            with open(self.schema_cache_path, "r") as f:
                self.schema = f.read()
        else:
            super().refresh_schema()
            Path(self.schema_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.schema_cache_path, "w") as f:
                f.write(self.schema)
