from llama_index import ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import Ollama

LLM_MODEL = "starling-lm"
EMBED_MODEL = "dmis-lab/biobert-v1.1"


def get_service_context():
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        embed_batch_size=32,
    )

    llm = Ollama(
        model=LLM_MODEL,
        base_url="http://ollama:11434",
        request_timeout=30.0,
        temperature=0.0,
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    return service_context
