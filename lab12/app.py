import os 
from dotenv import load_dotenv 

# ibm machine learning library
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 

# langchain
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.tools.render import render_text_description_and_args
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import RetrievalQA 

from langchain_huggingface import HuggingFaceEmbeddings

# langchain-community

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import BaseTool
from langchain_core.tools import Tool
from langchain.tools import tool
# Load API_KEY
from load_env import (
	credentials,
  project_id,
  tavily_api_key
)

from llm_model import create_llm
from prompt import (
	create_prompt
)
from tools import (
  load_tools,
  create_tools_info 
)

from agent import (
  create_agent,
  create_memory
)
# Define Tools


loader = PyPDFLoader("./data/SPRI_AI_Brief_2023년12월호.pdf")
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 100, chunk_overlap=100
)
documents = loader.load_and_split(text_splitter)
embedding = HuggingFaceEmbeddings()
vectordb = FAISS.from_documents(
	documents=documents,
 embedding=embedding
)

retriever = vectordb.as_retriever()
retriever_tool = create_retriever_tool(
	retriever, 
 name = "pdf_search",
 description="2023년 12월 AI 관련 정보를 PDF 문서에서 검색합니다. '2023년 12월 AI 산업동향' 과 관련된 질문은 이 도구를 사용해야 합니다!"
)

llm = create_llm(credentials, project_id)
qa = RetrievalQA.from_chain_type(
  llm = llm,
  chain_type = "stuff",
  retriever = vectordb.as_retriever()
)

tools = load_tools(retriever_tool)
#tools = tools.append(get_query)



prompt = create_prompt()
prompt = create_tools_info(prompt, tools)

memory = create_memory()

llm = create_llm(credentials, project_id)

chain = create_agent(prompt, llm, memory)

# Create an agent executor by passing in the agent and tools
# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정
agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory)

result = agent_executor.invoke({"input": "YouTube 2024년부터 AI 생성콘텐츠 표시 의무화에 대한 내용을 PDF 문서에서 알려줘"})

print(result['output'])