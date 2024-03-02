import logging
import time
from datetime import datetime, timedelta
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Sequence

from llama_index import (Document, KnowledgeGraphIndex, ServiceContext,
                         SimpleDirectoryReader, VectorStoreIndex)

from reader import AbstractCSVReader, FullArticleXMLReader
from service_context import get_service_context

LOG_PATH = Path(f"logs/{Path(__file__).stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
DATA_DIR = "/data/pmc-open-access-subset/6291"
PERSIST_DIR = Path("e")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

from transformers import pipeline

triplet_extractor = pipeline(
    'text2text-generation',
    model='Babelscape/rebel-large',
    tokenizer='Babelscape/rebel-large',
    device='cuda:0',
)


def extract_triplets(input_text):
    try:
        generated_token_ids = triplet_extractor(input_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]
        text = triplet_extractor.tokenizer.batch_decode([generated_token_ids])[0]
    except Exception:
        logger.exception(f"Failed to extract triplets from {input_text}")
        return []

    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and subject in input_text and relation != '' and relation in input_text and object_ != '' and object_ in input_text:
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return triplets


def main():
    start = time.time()
    service_context = get_service_context()

    # vector_store_index_perist_dir = PERSIST_DIR / "vector_store_index"
    # if not vector_store_index_perist_dir.exists():
    #     documents = read_documents()
    #     logger.info("Generating vector store index")
    #     generate_vector_store_index(service_context, documents, vector_store_index_perist_dir)

    graph_store_index_persist_dir = PERSIST_DIR / "graph_store_index"
    if not graph_store_index_persist_dir.exists():
        documents = read_documents()
        logger.info("Generating graph store index")
        generate_graph_store_index(service_context, documents, graph_store_index_persist_dir)

    end = time.time()
    logger.info(f"Total time: {timedelta(seconds=end - start)}")


@cache
def read_documents():
    logger.info("Reading documents")
    start = time.time()
    reader = SimpleDirectoryReader(
        DATA_DIR,
        required_exts=[".csv"],
        file_extractor={
            ".csv": AbstractCSVReader(),
            ".xml": FullArticleXMLReader(),
        },
    )
    documents = reader.load_data(show_progress=True)
    end = time.time()
    logger.info(f"Reading documents took {timedelta(seconds=end - start)}")
    return documents


def generate_graph_store_index(
    service_context: ServiceContext, documents: Sequence[Document], persist_dir: str | PathLike
):
    start = time.time()
    knowledge_graph_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=2,
        service_context=service_context,
        include_embeddings=True,
        show_progress=True,
        kg_triplet_extract_fn=extract_triplets,
    )
    end = time.time()
    logger.info(f"Generating graph store index took {timedelta(seconds=end - start)}")
    start = time.time()
    knowledge_graph_index.storage_context.persist(persist_dir=persist_dir)
    end = time.time()
    logger.info(f"Persisting graph store index took {timedelta(seconds=end - start)}")


def generate_vector_store_index(
    service_context: ServiceContext, documents: Sequence[Document], persist_dir: str | PathLike
):
    start = time.time()
    vector_store_index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        show_progress=True,
    )
    end = time.time()
    logger.info(f"Generating vector store index took {timedelta(seconds=end - start)}")
    start = time.time()
    vector_store_index.storage_context.persist(persist_dir=persist_dir)
    end = time.time()
    logger.info(f"Persisting vector store index took {timedelta(seconds=end - start)}")


if __name__ == "__main__":
    main()
