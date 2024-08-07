from fastapi import FastAPI  
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langserve import add_routes 

vectordb = FAISS.from_texts(
  ["cats like fish", "dogs like sticks"], embedding=HuggingFaceEmbeddings()
)

retriever = vectordb.as_retriever()

app = FastAPI(
  title="LangChain Server",
  version = "1.0",
  description="Spin up a simple api server using Langchain's Runnable interfaces"
)

add_routes(app, 
           retriever,
           path="/openai",
           )

