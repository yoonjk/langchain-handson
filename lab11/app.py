import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ibm import WatsonxLLM
from langchain_huggingface import HuggingFaceEmbeddings
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain import hub


load_dotenv()

credentials = {
	"apikey": os.getenv("API_KEY", None),
  "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)


# setup the tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def square(a) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

human_prompt = """{input}
    {agent_scratchpad}
    """
    
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a mathematical assistant.
        Use your tools to answer questions. If you do not have a tool to
        answer the question, say so. 

        Return only the answers. e.g
        Human: What is 1 + 1?
        AI: 2
        """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human_prompt),
    ]
)

# Choose the LLM that will drive the agent
params = {
	GenParams.DECODING_METHOD : "greedy", 
  GenParams.MIN_NEW_TOKENS: 1,
  GenParams.MAX_NEW_TOKENS: 100
}

llm = WatsonxLLM(
  			model_id = ModelTypes.LLAMA_2_70B_CHAT.value,
    		apikey = credentials['apikey'],
    		url = credentials['url'],
        params = params, 
        project_id = project_id
      )

# Create an LLMMathChain instance from the LLM, which is used to handle mathematical queries
math_chain = LLMMathChain.from_llm(llm=llm)

# setup the toolkit
tools = [add, multiply, square]

tool_names=", ".join([t.name for t in tools]),
print(tool_names)
# Construct the OpenAI Tools agent
memory = ConversationBufferMemory()

chain = ( RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        chat_history=lambda x: memory.chat_memory.messages,
    )
    | prompt | llm | ReActSingleInputOutputParser()

  )
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=False, handle_parsing_errors=True)

result = agent_executor.invoke({"input": "what is 1 + 1?"})

print(result['output'])