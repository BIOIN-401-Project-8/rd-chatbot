from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import YouTubeSearchTool

youtube = YouTubeSearchTool()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    # model_path="/root/.cache/huggingface/hub/models--TheBloke--Mixtral-8x7B-Instruct-v0.1-GGUF/snapshots/fa1d3835c5d45a3a74c0b68805fcdc133dba2b6a/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    model_path="/root/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=0,
    callback_manager=callback_manager,
    verbose=True,
)

prompt = PromptTemplate(
    template="""
You are a movie expert. You find movies from a genre or plot.

ChatHistory:{chat_history}
Question:{input}

Answer:
""",
    input_variables=["chat_history", "input"],
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_parser=StrOutputParser(),
    memory=memory,
    verbose=True,
)

tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=llm_chain.run,
        return_direct=True,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word 'trailer'. Return a link to a YouTube video.",
        func=youtube.run,
        return_direct=True
    )
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory,handle_parsing_errors=True)


while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
