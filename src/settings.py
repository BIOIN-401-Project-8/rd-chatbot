import httpx
import torch
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.utils import get_cache_dir
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoModel

EMBED_MODEL = "michiyasunaga/BioLinkBERT-base"
OLLAMA_MODEL = "starling-lm"


def configure_settings(callback_manager: CallbackManager | None = None):
    # Pulling the model with Ollama
    # TODO: display this as a progress bar
    httpx.post("http://ollama:11434/api/pull", json={"name": OLLAMA_MODEL}, timeout=600.0)
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        base_url="http://ollama:11434",
        request_timeout=30.0,
        temperature=0.0,
    )
    cache_folder = get_cache_dir()
    Settings.embed_model = HuggingFaceEmbedding(
        model=AutoModel.from_pretrained(EMBED_MODEL, cache_dir=cache_folder, torch_dtype=torch.float16),
        embed_batch_size=16,
    )
    Settings.num_output = 768
    Settings.callback_manager = callback_manager
