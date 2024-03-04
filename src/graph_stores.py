from typing import Any, Dict, List

from llama_index.graph_stores.neo4j import Neo4jGraphStore


class CustomNeo4jGraphStore(Neo4jGraphStore):
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "neo4j",
        node_label: str = "Entity",
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

        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map

        subjs = [subj.upper() for subj in subjs]

        query = (
            f"""MATCH p=(n1:`{self.node_label}`)-[*1..{depth}]->() """
            f"""{"WHERE apoc.coll.intersection(apoc.convert.toList(n1.N_Name), $subjs)" if subjs else ""} """
            "UNWIND relationships(p) AS rel "
            "WITH n1._N_Name AS subj, p, apoc.coll.flatten(apoc.coll.toSet("
            "collect([type(rel), rel.name, endNode(rel)._N_Name, endNode(rel)._I_GENE]))) AS flattened_rels "
            f"RETURN subj, collect(flattened_rels) AS flattened_rels LIMIT {limit}"
        )

        data = list(self.query(query, {"subjs": subjs}))
        if not data:
            return rel_map

        for record in data:
            # replace _ with space
            flattened_rels = list(
                set(
                    tuple(
                        [str(flattened_rel[1]) or str(flattened_rel[0]), str(flattened_rel[2]) or str(flattened_rel[3])]
                    )
                    for flattened_rel in record["flattened_rels"]
                )
            )
            rel_map[record["subj"]] = flattened_rels

        rel_map_organization = self.get_rel_map_organization(subjs, depth, limit)
        for subj, rels in rel_map_organization.items():
            if subj in rel_map:
                rel_map[subj] += rels
            else:
                rel_map[subj] = rels
        return rel_map

    def get_rel_map_organization(
        self, subjs: List[str] | None = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            return rel_map

        subjs = [subj.upper() for subj in subjs]

        query = f"""
            MATCH p=(n1:`{self.node_label}`)<-[:ORGANIZATION*1..{depth}]-(n2)
            {"WHERE apoc.coll.intersection(apoc.convert.toList(n1.N_Name), $subjs)" if subjs else ""}
            RETURN n1._N_Name AS _N_Name, n2.Address1 AS Address1, n2.Address2 AS Address2, n2.City AS City, n2.Country AS Country, n2.Email AS Email, n2.Fax as Fax, n2.Name as Name, n2.Phone as Phone, n2.State as State, n2.TollFree as TollFree, n2.URL as URL, n2.ZipCode as ZipCode
            LIMIT {limit}
        """
        organizations = list(self.query(query, {"subjs": subjs}))

        if not organizations:
            return rel_map

        for organization in organizations:
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
            for obj in organization["_N_Name"].split("|"):
                if obj not in rel_map:
                    rel_map[obj] = []
                rel_map[obj].append(
                    ("has organization", "\n".join(organization_description),)
                )
        return rel_map

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.
        """
        from pathlib import Path

        if Path("schema.txt").exists():
            with open("schema.txt", "r") as f:
                self.schema = f.read()
        else:
            super().refresh_schema()
            with open("schema.txt", "w") as f:
                f.write(self.schema)
