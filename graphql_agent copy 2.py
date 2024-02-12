# %%
from langchain import hub
from langchain.agents import (AgentExecutor, AgentType, create_react_agent,
                              initialize_agent, load_tools)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.tools import BaseTool, YouTubeSearchTool
from langchain_community.tools.graphql.tool import BaseGraphQLTool
from langchain_community.utilities.graphql import GraphQLAPIWrapper

# %%
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from llm2 import get_llm_chain

llm = get_llm_chain()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

example_input = """query {
  articles(where: {abstractText_CONTAINS: "Duchenne muscular dystrophy"}, options: {limit: 8}) {
    abstractText
    source
  }
}"""

graphql_wrapper = GraphQLAPIWrapper(
    graphql_endpoint="https://rdas.ncats.nih.gov/api/articles",
)

class AbstractTool(BaseGraphQLTool):
    name: str = "get_abstracts"

    def __init__(self, graphql_wrapper, **kwargs):
        super().__init__(graphql_wrapper=graphql_wrapper, **kwargs)

    def run(self, tool_input: str, **kwargs):
        query = f'''query {{
    articles(where: {{abstractText_CONTAINS: "{tool_input.strip()}"}}, options: {{limit: 8}}) {{
        abstractText
        pubmed_id
    }}
}}'''

tools = [
    AbstractTool(
        description=f"""Input to this tool is a rare disease name, output is abstracts from the RDAS API.
            Use this tool to query the RDAS API for abstracts for accurate information on a rare disease.
            To cite this information, use the JSON key which is the pubmed_id.
            Example Input: Duchenne muscular dystrophy""",
            graphql_wrapper=graphql_wrapper,
    )
]

# Assistant is able to cite sources and provide references for the information it provides, allowing users to verify the accuracy and reliability of the information they receive.
agent_prompt = PromptTemplate(
    input_variables=["agent_scratchpad", "chat_history", "input", "tool_names", "tools"],
    template="""GPT4 Correct User: Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Assistant is able to cite sources and provide references for the information it provides, allowing users to verify the accuracy and reliability of the information they receive. Citations should be at the end of the line in the format [pubmed_id].

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}<|end_of_turn|>GPT4 Correct Assistant:""",
)

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    verbose=True,
)

while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])


# What are the phenotypes of Hereditary spastic paraplegia?

# %%
