
from langchain.agents import AgentExecutor

from load_env import (
  credentials,
  project_id
)

from prompt import (
  create_prompt
)

from tools import (
  load_tools,
  create_tools_info 
)

from llm_model import (
  create_llm
)

from agent import (
  create_agent,
  create_memory
)

tools = load_tools()
prompt = create_prompt()
prompt = create_tools_info(prompt, tools)

memory = create_memory()

llm = create_llm(credentials, project_id)

chain = create_agent(prompt, llm, memory)

# Create an agent executor by passing in the agent and tools
# AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정
agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory)

result = agent_executor.invoke({"input": "판교 카카오 프렌즈샵 아지트점의 전화번호는 무엇인가요?"})

print(result['output'])