from itertools import chain
from typing import Any, Dict, List

from llama_index.graph_stores.neo4j import Neo4jGraphStore

from textualize import textualize_organizations, textualize_phenotypes, textualize_prevelances, textualize_rels


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

        rel_map_rel = self.get_rel_map_rel(subjs, depth, limit)
        rel_map_organization = self.get_rel_map_organization(subjs, limit)
        rel_map_phenotype = self.get_rel_map_phenotype(subjs, limit)
        rel_map_prevalence = self.get_rel_map_prevalence(subjs, limit)
        for subj, rels in chain(
            rel_map_rel.items(), rel_map_organization.items(), rel_map_phenotype.items(), rel_map_prevalence.items()
        ):
            if subj in rel_map:
                rel_map[subj] += rels
            else:
                rel_map[subj] = rels
        return rel_map

    def get_rel_map_rel(
        self, subjs: List[str] | None = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        if subjs is None or len(subjs) == 0:
            return {}
        # TODO: restore depth functionality
        query = f"""MATCH p=(n:`{self.node_label}`)-[r:R_rel]->(m)
            {"WHERE apoc.coll.intersection(apoc.convert.toList(n.N_Name), $subjs)" if subjs else ""}
            RETURN n._N_Name AS n__N_Name, n._I_GENE AS n__I_GENE, m._N_Name AS m__N_Name, m._I_GENE AS m__I_GENE, r.citations AS r_citations, r.interpretation AS r_interpretation, r.name AS r_name, r.value AS r_value
        """

        rels = list(self.query(query, {"subjs": subjs}))
        if not rels:
            return {}

        return textualize_rels(rels)

    def get_rel_map_organization(self, subjs: List[str] | None = None, limit: int = 30) -> Dict[str, List[List[str]]]:
        if subjs is None or len(subjs) == 0:
            return {}

        subjs = [subj.upper() for subj in subjs]

        query = f"""
            MATCH p=(m:`{self.node_label}`)<-[:ORGANIZATION]-(n)
            {"WHERE apoc.coll.intersection(apoc.convert.toList(m.N_Name), $subjs)" if subjs else ""}
            RETURN m._N_Name AS m__N_Name, m._I_CODE AS m__I_CODE, n.Address1 AS n_Address1, n.Address2 AS n_Address2, n.City AS n_City, n.Country AS n_Country, n.Email AS n_Email, n.Fax as n_Fax, n.Name as n_Name, n.Phone as n_Phone, n.State as n_State, n.TollFree as n_TollFree, n.URL as n_URL, n.ZipCode as n_ZipCode
            LIMIT {limit}
        """
        organizations = list(self.query(query, {"subjs": subjs}))

        if not organizations:
            return {}

        return textualize_organizations(organizations)

    def get_rel_map_phenotype(self, subjs: List[str] | None = None, limit: int = 30):
        if subjs is None or len(subjs) == 0:
            return {}

        subjs = [subj.upper() for subj in subjs]

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

    def get_rel_map_prevalence(self, subjs: List[str] | None = None, limit: int = 30) -> Dict[str, List[List[str]]]:
        if subjs is None or len(subjs) == 0:
            return {}

        subjs = [subj.upper() for subj in subjs]

        query = f"""
            MATCH p=(m:`{self.node_label}`)<-[r:PREVALENCE*1]-(n)
            {"WHERE apoc.coll.intersection(apoc.convert.toList(m.N_Name), $subjs)" if subjs else ""}
            RETURN m._N_Name AS m__N_Name, n.PrevalenceClass AS n_PrevalenceClass, n.PrevalenceGeographic AS n_PrevalenceGeographic, n.PrevalenceQualification AS n_PrevalenceQualification, n.PrevalenceValidationStatus AS n_PrevalenceValidationStatus, n.Source AS n_Source, n.ValMoy AS n_ValMoy
            LIMIT {limit}
        """
        prevalences = list(self.query(query, {"subjs": subjs}))

        if not prevalences:
            return {}

        return textualize_prevelances(prevalences)

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
