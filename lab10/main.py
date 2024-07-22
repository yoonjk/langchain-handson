from langchain.prompts import (
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain import hub
from langchain.tools import tool
from langchain.agents import AgentExecutor, load_tools, Tool, create_openai_tools_agent
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes 

from langchain_ibm import WatsonxLLM
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
import os 
from datetime import datetime


def get_function_tools():
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



def acgent_action():
  
  llm = create_llm()
  template = """
  Question: {question}
  """
  handler = StdOutCallbackHandler()
  config = {
    'callbacks' : [handler]
  }
  tools = [get_todays_date]
  # prompt = ChatPromptTemplate.from_template(template)
  prompt = ChatPromptTemplate.from_messages(
    [

        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # 중간 과정 전달
    ]
)
  agent =  prompt | llm
  #agent = prompt | llm

  memory = ConversationBufferMemory()
  agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory)
  agent_executor.invoke({"input": "테슬라  주인은 누구인지 자세히 알려줘?"}) 
  


acgent_action()



