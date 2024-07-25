from langchain.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder
)
from langchain.tools.render import render_text_description_and_args
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain import hub
from langchain.tools import tool
from langchain.agents import AgentExecutor, load_tools, Tool, create_openai_tools_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes 
from langchain_core.runnables import RunnablePassthrough

from langchain_ibm import WatsonxLLM
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
import os 
from datetime import datetime

@tool
def get_function_tools():
  """get_function_tools search."""
  search = TavilySearchAPIWrapper()
  tavily_tool = TavilySearchResults(api_wrapper=search)

  print(tavily_tool.name)


  

load_dotenv() 

credentials = {
  "apikey": os.getenv("API_KEY", None),
  "url" : "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY", None)

def create_llm():
  params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 300
  }
  
  llm = WatsonxLLM(
    model_id = ModelTypes.LLAMA_2_70B_CHAT.value,
    apikey = credentials['apikey'],
    url = credentials['url'],
    params=params, 
    project_id = project_id
  )
  
  return llm 

date = ''

@tool
def get_todays_date():
    """Get today's date in YYYY-MM-DD format."""
    global date
    date = datetime.now().strftime("%Y-%m-%d")
    return date


@tool
def add(a: int, b: int) -> int:
 """Adds two numbers together""" # this docstring gets used as the description
 return a + b # the actions our tool performs



def agent_action():
  llm = create_llm()
  tools = [get_function_tools, get_todays_date]
  human_prompt = """{input}
    {agent_scratchpad}
    (reminder to always respond in a JSON blob) """

  prompt = ChatPromptTemplate.from_messages(
    [

        MessagesPlaceholder("chat_history", optional=True),
        ("human", human_prompt),
    ]
  )
  prompt = prompt.partial(
    tools=render_text_description_and_args(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
  )
  memory = ConversationBufferMemory()

  chain = ( RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        chat_history=lambda x: memory.chat_memory.messages,
    )
    | prompt | llm | JSONAgentOutputParser()

  )

  agent_executor = AgentExecutor(agent=chain, tools=tools, handle_parsing_errors=True, verbose=True, memory=memory)
  #result = agent.invoke({"input": "테슬라  주인은 누구인지 자세히 알려줘?"}) 
  # result = agent_executor.invoke({"input": "테슬라  주인은 누구인지 자세히 알려줘?"})
  return agent_executor

action = agent_action()
print( action.invoke({"input": "테슬라  주인은 누구인지 자세히 알려줘?"}))
action.invoke({"input": "What is today's date?"})



