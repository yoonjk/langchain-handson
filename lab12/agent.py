
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser

def create_memory():
    memory = ConversationBufferMemory()
  
    return memory 
  
def create_agent(prompt, llm, memory):
  chain = ( RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        chat_history=lambda x: memory.chat_memory.messages,
    )
    | prompt | llm | JSONAgentOutputParser()
  )
  
  return chain 