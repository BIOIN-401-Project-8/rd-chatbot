import json

from neo4j import GraphDatabase, RoutingControl


def detach_delete(driver):
    driver.execute_query(
        "MATCH (n) DETACH DELETE n",
        database_="neo4j", routing_=RoutingControl.WRITE,
    )

with GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "12345678")) as driver:
    if input("Press y to continue...") == "y":
        detach_delete(driver)
