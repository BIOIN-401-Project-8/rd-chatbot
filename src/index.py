import logging
import time
from datetime import datetime, timedelta
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Sequence

from llama_index import Document, KnowledgeGraphIndex, ServiceContext, SimpleDirectoryReader, VectorStoreIndex

from reader import AbstractCSVReader, FullArticleXMLReader
from service_context import get_service_context

LOG_PATH = Path(f"logs/{Path(__file__).stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
DATA_DIR = "/data/pmc-open-access-subset/6291"
PERSIST_DIR = Path("/data/rgd-chatbot/storage/6291/")

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


def main():
    start = time.time()
    service_context = get_service_context()

    vector_store_index_perist_dir = PERSIST_DIR / "vector_store_index"
    if not vector_store_index_perist_dir.exists():
        documents = read_documents()
        logger.info("Generating vector store index")
        generate_vector_store_index(service_context, documents, vector_store_index_perist_dir)

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
