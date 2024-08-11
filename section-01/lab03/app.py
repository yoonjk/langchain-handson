from load_env import (
  credentials,
  project_id
)

from llm_model import create_llm

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "my-first"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_TRACING_V2 = True

def load_and_splitter(pdf_file_path):
  loader = PyPDFLoader(pdf_file_path)
  
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n"])
  spliter_docs = loader.load_and_split(text_splitter=text_splitter)

  return spliter_docs

def summarize_pdf(docs, llm):
  chain = load_summarize_chain(llm = llm, chain_type="map_reduce")
  
  summary = chain.invoke(docs)
  
  return summary

def embeddings_retriever(docs, embeddings):
  """vectore store embedding"""
  retriever = FAISS.from_documents(documents=docs, embedding=embeddings)
  
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
print(summarize['output_text'])

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

