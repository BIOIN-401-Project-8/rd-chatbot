import httpx
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.ollama import Ollama

from embeddings import SentenceTransformerEmbeddings

EMBED_MODEL = "intfloat/e5-base-v2"
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
    Settings.embed_model = SentenceTransformerEmbeddings(
        model_name_or_path=EMBED_MODEL,
        embed_batch_size=16,
    )
    Settings.num_output = 768
    Settings.callback_manager = callback_manager
