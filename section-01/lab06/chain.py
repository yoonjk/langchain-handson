
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def create_chain(llm, retriever, prompt, format_docs):
  chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain
  

def create_prompt_template(prompt):
    prompt_template= ChatPromptTemplate.from_template(prompt)
    
    return prompt_template