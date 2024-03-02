from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.llms.llamacpp import LlamaCpp

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    # model_path="/root/.cache/huggingface/hub/models--TheBloke--Mixtral-8x7B-Instruct-v0.1-GGUF/snapshots/fa1d3835c5d45a3a74c0b68805fcdc133dba2b6a/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    model_path="/root/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=0,
    callback_manager=callback_manager,
    verbose=True,
    temperature=0
)

tools = load_tools(
    ["graphql"],
    graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index",
)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

graphql_fields = """
allFilms {
    films {
        title
        director
        releaseDate
        speciesConnection {
            species {
                name
                classification
                homeworld {
                    name
                }
            }
        }
    }
}"""

suffix = """Search for the release dates of all the star wars films.

Instructions:
Do NOT make syntax errors.

Example action input:
query {
    allFilms {
        films {
            title
        }
    }
}

GraphQL schema:"""


agent.run(suffix + graphql_fields)
