from load_env import (
  credentials,
  project_id
)

import os 

from llm_model import create_llm

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

def load_and_splitter(pdf_file_path):
  loader = PyPDFLoader(pdf_file_path)
  
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n"])
  splitter_docs = loader.load_and_split(text_splitter=text_splitter)

  return splitter_docs

def summarize_pdf(docs, llm):
  chain = load_summarize_chain(llm = llm, chain_type="stuff")
  
  summary = chain.invoke(docs)
  
  return summary

def embeddings_retriever(docs, embeddings):
  """vectore store embedding"""
  retriever = FAISS.from_documents(documents=docs, embedding = embeddings)
  
  return retriever.as_retriever()

def format_docs(docs):
  """context documents"""
  format_docs = "\n\n".join([doc.page_content for doc in docs])
  
  return format_docs

def create_chain(llm, retriever, prompt):
  chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
  )

  
  return chain
  
llm = create_llm(credentials=credentials, project_id=project_id)
docs = load_and_splitter("./data/LangChain.pdf")
summarize = summarize_pdf(docs, llm)
print(summarize)

prompt_template = """
{context}
question : {question}
"""

embeddings = HuggingFaceEmbeddings()
retriever = embeddings_retriever(docs = docs, embeddings=embeddings)
prompt = ChatPromptTemplate.from_template(prompt_template)

chain = create_chain(llm = llm, retriever=retriever, prompt=prompt)

result = chain.invoke("langchain에 대해 설명해줘. You answer the korea language")

print(result)

