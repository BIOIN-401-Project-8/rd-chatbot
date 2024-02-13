# %%
import json
import logging
from datetime import datetime
from pathlib import Path
import time
from neo4j import GraphDatabase, RoutingControl
LOG_PATH = Path(f"logs/{Path(__file__).stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")


logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(LOG_PATH),
    ],
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger()

LIMIT = -1


def write_node(driver, node):
    n = node["n"]
    labels = ":".join(f"`{label}`" for label in n.labels)
    properties = "{" + ", ".join(f"`{key}`: {json.dumps(value)}" for key, value in n.items()) + "}"
    query = f"CREATE (n:{labels} {properties})"
    logger.debug(query)
    driver.execute_query(
        query,
        database_="neo4j",
        routing_=RoutingControl.WRITE,
    )


def write_nodes(driver, nodes):
    for node in nodes:
        write_node(driver, node)


def write_relationship(driver, relationship):
    r = relationship["r"]
    properties = "{" + ", ".join(f"`{key}`: {json.dumps(value)}" for key, value in r.items()) + "}"
    query = f"""
        MATCH (n) WHERE ID(n) = {r.start_node.element_id}
        MATCH (m) WHERE ID(m) = {r.end_node.element_id}
        CREATE (n)-[r:{r.type} {properties}]->(m)
    """
    logger.debug(query)
    driver.execute_query(
        query,
        database_="neo4j",
        routing_=RoutingControl.WRITE,
    )


def write_relationships(driver, relationships):
    for relationship in relationships:
        write_relationship(driver, relationship)


def read_nodes(driver):
    query = """
        MATCH (n)
        RETURN n
    """
    if LIMIT > 0:
        query += f"""
        LIMIT {LIMIT}
        """
    logger.debug(query)
    records, _, _ = driver.execute_query(
        query,
        database_="neo4j",
        routing_=RoutingControl.READ,
    )
    return records


def read_relationships(driver):
    query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
    """
    if LIMIT > 0:
        query += f"""
        LIMIT {LIMIT}
        """
    logger.debug(query)
    records, _, _ = driver.execute_query(
        query,
        database_="neo4j",
        routing_=RoutingControl.READ,
    )
    return records


# %%
with GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "12345678")) as driver, GraphDatabase.driver(
    "neo4j+ssc://disease.ncats.io:7687", auth=("", "")
) as driver_disease_ncats_io:
    logging.info("Reading nodes from disease.ncats.io")
    start = time.time()
    nodes = read_nodes(driver_disease_ncats_io)
    end = time.time()
    logging.info(f"Time to read nodes from disease.ncats.io: {end - start}")
    logging.info("Writing nodes to local")
    start = time.time()
    write_nodes(driver, nodes)
    end = time.time()
    logging.info(f"Time to write nodes to local: {end - start}")
    logging.info("Reading relationships from disease.ncats.io")
    start = time.time()
    relationships = read_relationships(driver_disease_ncats_io)
    end = time.time()
    logging.info(f"Time to read relationships from disease.ncats.io: {end - start}")
    logging.info("Writing relationships to local")
    start = time.time()
    write_relationships(driver, relationships)
    end = time.time()


# # %%
# with GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "12345678")) as driver:
#     records = read_relationships(driver)
#     for record in records:
#         print(record)
