import httpx
import torch
from llama_index.core import ServiceContext
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.utils import get_cache_dir
from transformers import AutoModel

EMBED_MODEL = "michiyasunaga/BioLinkBERT-base"
EMBED_DIM = 768
OLLAMA_MODEL = "starling-lm"


def get_service_context(callback_manager: CallbackManager = None):
    cache_folder = get_cache_dir()
    embed_model = HuggingFaceEmbedding(
        model=AutoModel.from_pretrained(EMBED_MODEL, cache_dir=cache_folder, torch_dtype=torch.float16),
        embed_batch_size=16,
    )

    httpx.post("http://ollama:11434/api/pull", json={"name": OLLAMA_MODEL}, timeout=600.0)
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url="http://ollama:11434",
        request_timeout=30.0,
        temperature=0.0,
    )

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm, callback_manager=callback_manager)
    return service_context
