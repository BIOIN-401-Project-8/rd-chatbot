import argparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (DirectoryLoader,
                                                  UnstructuredHTMLLoader)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
DATA_PATH = "/data/pmc-open-access-subset/articles_Carpenter syndrome/"
VECTORSTORE_PATH = "/data/rgd-chatbot/vectorstore/faiss"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--vectorstore_path", type=str, default=VECTORSTORE_PATH)
    args = parser.parse_args()

    loader = DirectoryLoader(
        args.data_path,
        glob="*.xml",
        loader_cls=UnstructuredHTMLLoader,
        show_progress=True
    )

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(args.vectorstore_path)


if __name__ == "__main__":
    main()
