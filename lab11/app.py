import os
from dotenv import load_dotenv
from datetime import datetime 

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ibm import WatsonxLLM
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents import AgentExecutor, Tool
from langchain.tools.render import render_text_description_and_args
from langchain.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.agent_toolkits.load_tools import load_tools


# machine learning library
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 

import nasapy

load_dotenv()

credentials = {
	"apikey": os.getenv("API_KEY", None),
  "url": "https://us-south.ml.cloud.ibm.com"
}
tavily_api_key= os.getenv("TAVILY_API_KEY", None)
google_api_key = os.getenv("GOOGLE_API_KEY", None)
google_cse_id = os.getenv("GOOGLE_CSE_ID", None)

os.environ['GOOGLE_API_KEY'] = google_api_key
os.environ['GOOGLE_CSE_ID'] = google_cse_id
os.environ['TAVILY_API_KEY'] = tavily_api_key
project_id = os.getenv("PROJECT_ID", None)

nasa_key = os.getenv("NASA_KEY", None)
n = nasapy.Nasa(key=nasa_key)

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
def divide(a: int, b: int) -> int:
    """Divide two numbers"""

    return a / b 

date = ''

@tool
def get_todays_date():
    """Get today's date in YYYY-MM-DD format."""
    global date
    date = datetime.now().strftime("%Y-%m-%d")
    return date

@tool(return_direct=True)
def get_astronomy_image():
    """Get NASA's Astronomy Picture of the Day today."""
    global date
    apod = n.picture_of_the_day(date, hd=True)
    return apod['url']


tavily_tool = TavilySearchResults()

# Define tool
tools = [get_todays_date, add, multiply, get_astronomy_image, divide, tavily_tool]
tools.extend(load_tools(['wikipedia']))


# Choose the LLM that will drive the agent
params = {
	GenParams.DECODING_METHOD : "greedy", 
  GenParams.MIN_NEW_TOKENS: 1,
  GenParams.MAX_NEW_TOKENS: 100,
  GenParams.STOP_SEQUENCES : ['\nObservation', '\n\n']
}

llm = WatsonxLLM(
  model_id = 'meta-llama/llama-3-70b-instruct',
  apikey = credentials['apikey'],
  url = credentials['url'],
  params = params, 
  project_id = project_id
)


# Create an LLMMathChain instance from the LLM, which is used to handle mathematical queries

system_prompt = """Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:"
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Follow this format:
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""

human_prompt = """{input}
{agent_scratchpad}
(reminder to always respond in a JSON blob)"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
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
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory)

result = agent_executor.invoke({"input": "What is dbama's first name?"})

print(result['output'])