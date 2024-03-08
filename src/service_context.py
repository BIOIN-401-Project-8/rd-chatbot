import httpx
from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager
from llama_index.llms import Ollama

from embeddings import SentenceTransformerEmbeddings

OLLAMA_MODEL = "starling-lm"


def get_service_context(callback_manager: CallbackManager = None):
    embed_model = SentenceTransformerEmbeddings(
        model_name="intfloat/e5-large-v2",
        embed_batch_size=1,
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
