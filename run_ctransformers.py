from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# llm = CTransformers(model="./miqu-1-70b.q4_k_m.gguf",
#                      callbacks=[StreamingStdOutCallbackHandler()],
#                     gpu_layers=32,
# )
from timeit import default_timer as timer

from ctransformers import AutoModelForCausalLM

start = timer()
llm = AutoModelForCausalLM.from_pretrained(
  "TheBloke/Mistral-7B-v0.1-GGUF",
  model_file="mistral-7b-v0.1.Q4_K_M.gguf",
  model_type="mistral",
  gpu_layers=-1,
)
end = timer()
print(f"Load time taken: {end - start}")

for i in range(2):
  start = timer()
  response = llm("What is CK Syndrome?")
  end = timer()

  print(f"Response time taken: {end - start}")
  print(response)
