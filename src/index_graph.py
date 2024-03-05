# %%
import os
from typing import List, Iterable
import torch
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer


def batch(iterable: Iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for batch_texts in batch(texts, self.batch_size):
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512).to(self.model.device)
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            batch_embeddings = last_hidden_states.mean(dim=1).tolist()
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed search query."""
        return self.embed_documents([text])[0]


node_label = "S_PHENOTYPE"
# embeddings = CustomEmbeddings(model_name="michiyasunaga/BioLinkBERT-large", batch_size=8)
# embeddings = CustomEmbeddings(model_name="dmis-lab/biobert-v1.1", batch_size=8)
# model_name = "all-MiniLM-L6-v2"
model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs = {'batch_size': 8})


#%%
def test_custom_embeddings():
    texts = [
        "\n_N_Name:GRACILE SYNDROME|FELLMAN DISEASE|FELLMAN SYNDROME|FINNISH LACTIC ACIDOSIS WITH HEPATIC HEMOSIDEROSIS|FINNISH LETHAL NEONATAL METABOLIC SYNDROME|FLNMS|GROWTH DELAY-AMINOACIDURIA-CHOLESTASIS-IRON OVERLOAD-LACTIC ACIDOSIS-EARLY DEATH SYNDROME|GROWTH RESTRICTION-AMINOACIDURIA-CHOLESTASIS-IRON OVERLOAD-LACTIC ACIDOSIS-EARLY DEATH SYNDROME|GROWTH RETARDATION, AMINOACIDURIA, CHOLESTASIS, IRON OVERLOAD, LACTIC ACIDOSIS AND EARLY DEATH",
        "\n_N_Name:ABETALIPOPROTEINEMIA|ABETALIPOPROTEINEMIA NEUROPATHY|ABL|APOLIPOPROTEIN B DEFICIENCY|BASSEN KORNZWEIG SYNDROME|BASSEN-KORNZWEIG DISEASE|BETALIPOPROTEIN DEFICIENCY DISEASE|CONGENITAL BETALIPOPROTEIN DEFICIENCY SYNDROME|HOMOZYGOUS FAMILIAL HYPOBETALIPOPROTEINEMIA|MICROSOMAL TRIGLYCERIDE TRANSFER PROTEIN DEFICIENCY|MICROSOMAL TRIGLYCERIDE TRANSFER PROTEIN DEFICIENCY DISEASE|MTP DEFICIENCY",
    ]
    return embeddings.embed_documents(texts)

e = test_custom_embeddings()
e, len(e[0])
#%%

from typing import Any, List, Optional, Type
from langchain_community.vectorstores.neo4j_vector import SearchType, DEFAULT_SEARCH_TYPE
class CustomNeo4jVector(Neo4jVector):
    @classmethod
    def from_existing_graph(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        node_label: str,
        embedding_node_property: str,
        text_node_properties: List[str],
        *,
        keyword_index_name: Optional[str] = "keyword",
        index_name: str = "vector",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Initialize and return a Neo4jVector instance from an existing graph.

        This method initializes a Neo4jVector instance using the provided
        parameters and the existing graph. It validates the existence of
        the indices and creates new ones if they don't exist.

        Returns:
        Neo4jVector: An instance of Neo4jVector initialized with the provided parameters
                    and existing graph.

        Example:
        >>> neo4j_vector = Neo4jVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

        Note:
        Neo4j credentials are required in the form of `url`, `username`, and `password`,
        and optional `database` parameters passed as additional keyword arguments.
        """
        # Validate the list is not empty
        if not text_node_properties:
            raise ValueError(
                "Parameter `text_node_properties` must not be an empty list"
            )
        # Prefer retrieval query from params, otherwise construct it
        if not retrieval_query:
            retrieval_query = (
                f"RETURN reduce(str='', k IN {text_node_properties} |"
                " str + '\\n' + k + ': ' + coalesce(node[k], '')) AS text, "
                "node {.*, `"
                + embedding_node_property
                + "`: Null, id: Null, "
                + ", ".join([f"`{prop}`: Null" for prop in text_node_properties])
                + "} AS metadata, score"
            )
        store = cls(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            retrieval_query=retrieval_query,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            **kwargs,
        )

        # Check if the vector index already exists
        embedding_dimension = store.retrieve_existing_index()

        # If the vector index doesn't exist yet
        if not embedding_dimension:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                f"Index with name {store.index_name} already exists."
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )
        # FTS index for Hybrid search
        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index(text_node_properties)
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index(text_node_properties)
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Populate embeddings
        while True:
            fetch_query = (
                f"MATCH (n:`{node_label}`) "
                f"WHERE n.{embedding_node_property} IS null "
                "AND any(k in $props WHERE n[k] IS NOT null) "
                f"RETURN elementId(n) AS id, reduce(str='',"
                "k IN $props | str + '\\n' + k + ':' + coalesce(n[k], '')) AS text "
                "LIMIT 1000"
            )
            data = store.query(fetch_query, params={"props": text_node_properties})
            # HACK: Remove the prefix from the embedding
            text_embeddings = embedding.embed_documents([el["text"].removeprefix("\n_N_Name:") for el in data])

            params = {
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(data, text_embeddings)
                ]
            }

            store.query(
                "UNWIND $data AS row "
                f"MATCH (n:`{node_label}`) "
                "WHERE elementId(n) = row.id "
                f"CALL db.create.setVectorProperty(n, "
                f"'{embedding_node_property}', row.embedding) "
                "YIELD node RETURN count(*)",
                params=params,
            )
            # If embedding calculation should be stopped
            if len(data) < 1000:
                break
        return store

model_name_clean = model_name.split("/")[-1].replace("-", "_")
# %%
vector_index = CustomNeo4jVector.from_existing_graph(
    embeddings,
    url="bolt://neo4j:7687",
    username="neo4j",
    password=os.environ["NEO4J_PASSWORD"],
    index_name=f"node_vector_index_{node_label}_{model_name_clean}",
    node_label=node_label,
    text_node_properties=["_N_Name"],
    embedding_node_property=f"embedding_{model_name_clean}"
)

# %%
texts = [
    "DUCHENNE MUSCULAR DYSTROPHY",
    "DMD"
]
for i, text in enumerate(texts):
    with open(f"embedding_{text.replace(' ', '_')}_{model_name_clean}.json", "w") as f:
        f.write(
            str(embeddings.embed_query(text))
        )

# %%
