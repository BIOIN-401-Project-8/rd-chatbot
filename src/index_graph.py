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
embeddings = CustomEmbeddings(model_name="dmis-lab/biobert-v1.1", batch_size=8)


#%%
def test_custom_embeddings():
    texts = [
        "\n_N_Name:GRACILE SYNDROME|FELLMAN DISEASE|FELLMAN SYNDROME|FINNISH LACTIC ACIDOSIS WITH HEPATIC HEMOSIDEROSIS|FINNISH LETHAL NEONATAL METABOLIC SYNDROME|FLNMS|GROWTH DELAY-AMINOACIDURIA-CHOLESTASIS-IRON OVERLOAD-LACTIC ACIDOSIS-EARLY DEATH SYNDROME|GROWTH RESTRICTION-AMINOACIDURIA-CHOLESTASIS-IRON OVERLOAD-LACTIC ACIDOSIS-EARLY DEATH SYNDROME|GROWTH RETARDATION, AMINOACIDURIA, CHOLESTASIS, IRON OVERLOAD, LACTIC ACIDOSIS AND EARLY DEATH",
        "\n_N_Name:ABETALIPOPROTEINEMIA|ABETALIPOPROTEINEMIA NEUROPATHY|ABL|APOLIPOPROTEIN B DEFICIENCY|BASSEN KORNZWEIG SYNDROME|BASSEN-KORNZWEIG DISEASE|BETALIPOPROTEIN DEFICIENCY DISEASE|CONGENITAL BETALIPOPROTEIN DEFICIENCY SYNDROME|HOMOZYGOUS FAMILIAL HYPOBETALIPOPROTEINEMIA|MICROSOMAL TRIGLYCERIDE TRANSFER PROTEIN DEFICIENCY|MICROSOMAL TRIGLYCERIDE TRANSFER PROTEIN DEFICIENCY DISEASE|MTP DEFICIENCY",
    ]
    return embeddings.embed_documents(texts)

e = test_custom_embeddings()
e, len(e[0])
# %%
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    url="bolt://neo4j:7687",
    username="neo4j",
    password=os.environ["NEO4J_PASSWORD"],
    index_name=f"node_vector_index_{node_label}_biobert_v1_1",
    node_label=node_label,
    text_node_properties=["_N_Name"],
    embedding_node_property="embedding_biobert_v1_1",
)

# %%

with open("embedding", "w") as f:
    f.write(
        str(embeddings.embed_query("\n_N_Name:DUCHENNE MUSCULAR DYSTROPHY"))
    )

# %%
