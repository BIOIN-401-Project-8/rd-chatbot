from typing import Any, List

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()
    _embed_batch_size: int = PrivateAttr()

    def __init__(
        self,
        model_name_or_path: str = 'intfloat/e5-large-v2',
        embed_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self._model = SentenceTransformer(model_name_or_path)
        self._embed_batch_size = embed_batch_size
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerEmbeddings"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._get_text_embeddings([query])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._get_text_embeddings([text])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, batch_size=self._embed_batch_size).tolist()
        return embeddings
