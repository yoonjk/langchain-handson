from load_env import (
  credentials,
  project_id
)

import os 

from llm_model import create_llm
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

def load_and_splitter(pdf_file_path):
  loader = PyPDFLoader(pdf_file_path)
  
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10, separators=["\n"])
  splitter_docs = loader.load_and_split(text_splitter=text_splitter)

  return splitter_docs

def summarize_pdf(docs, llm):
  chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
  
  summary = chain.invoke(docs)
  
  return summary

def embeddings_retriever(docs, embeddings):
  """vectore store embedding"""
  retriever = FAISS.from_documents(documents=docs, embedding = embeddings)
  
  return retriever

def format_docs(docs):
  """context documents"""
  format_docs = "\n\n".join([doc.page_content for doc in docs])
  
  return format_docs


  
llm = create_llm(credentials=credentials, project_id=project_id)
docs = load_and_splitter("./data/langchain.pdf")
summary = summarize_pdf(docs = docs, llm=llm)
embeddings = HuggingFaceEmbeddings()

vectordb = embeddings_retriever(docs, embeddings=embeddings)
print(summary['output_text'])

query_vector = "Agent란 무엇인가. you answer the korea language."

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

result = retriever.invoke(query_vector)

print(result)
# Print the most similar documents


