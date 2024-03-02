from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# llm = LlamaCpp(
#     # model_path="/root/.cache/huggingface/hub/models--TheBloke--Mixtral-8x7B-Instruct-v0.1-GGUF/snapshots/fa1d3835c5d45a3a74c0b68805fcdc133dba2b6a/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
#     # model_path="/root/.cache/huggingface/hub/models--miqudev--miqu-1-70b/snapshots/cfc21484a813e610b5e0471d955f5d54001bd0b0/miqu-1-70b.q4_k_m.gguf",
#     # model_path="/root/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
#     model_path="/root/.cache/huggingface/hub/models--TheBloke--Starling-LM-7B-alpha-GGUF/snapshots/028036db8fcc83779056e6f43d803cd2b0217add/starling-lm-7b-alpha.Q4_K_M.gguf",
#     n_ctx=4096,
#     n_gpu_layers=0,
#     callback_manager=callback_manager,
#     verbose=True,
#     temperature=0
# )
llm = CTransformers(
    model="TheBloke/Starling-LM-7B-alpha-GGUF",
    model_file="starling-lm-7b-alpha.Q4_K_M.gguf",
    model_type="mistral",
    max_new_tokens=8192,
    context_length=8192,
    temperature=0,
    gpu_layers=-1,
    verbose=True,
)
prompt = PromptTemplate(
    template="""GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:""",
    input_variables=["prompt"],
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = load_tools(
    ["graphql"],
    graphql_endpoint="https://rdas.ncats.nih.gov/api/articles",
)

agent = initialize_agent(
    tools, llm_chain, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

graphql_fields = '''
articles {
    abstractText
    citedByCount
    omim_evidence
    pubmed_id
    source
    title
}
'''

suffix = """
Search for the phenotypes of Hereditary spastic paraplegia

Instructions:
Do NOT make syntax errors.
Use a limit of 8.

Example action input:
query {
  articles(where: {abstractText_CONTAINS: "Duchenne muscular dystrophy"}, options: {limit: 8}) {
    abstractText
  }
}

GraphQL schema:"""


agent.run(suffix + graphql_fields)
