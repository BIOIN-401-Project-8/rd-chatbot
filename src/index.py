import logging
import os.path
import sys
from pathlib import Path

from llama_index import (SimpleDirectoryReader, StorageContext,
                         VectorStoreIndex, download_loader,
                         load_index_from_storage)
from llama_index.embeddings import resolve_embed_model

from reader import AbstractCSVReader, FullArticleXMLReader
from service_context import EMBED_MODEL, get_service_context

DATA_DIR = "/data/pmc-open-access-subset/6291"
PERSIST_DIR = f"/data/rgd-chatbot/storage/6291/vector_store_index/{EMBED_MODEL}"


def main():
    service_context = get_service_context()

    if not os.path.exists(PERSIST_DIR):
        reader = SimpleDirectoryReader(
            DATA_DIR,
            file_extractor={
                ".csv": AbstractCSVReader(),
                ".xml": FullArticleXMLReader(),
            },
        )
        documents = reader.load_data(show_progress=True)
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(
            storage_context,
            service_context=service_context,
            show_progress=True,
        )
        # append new data?

if __name__ == "__main__":
    main()
