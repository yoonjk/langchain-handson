from fastapi import FastAPI, UploadFile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from load_env import (
  credentials,
  project_id
)
from llm_model import create_llm
from load_doc import (
  load_db,
  create_stuff_docs_chain
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes 
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser

import os 

from load_doc import (
  load_and_split,
  embedding_vector
)

app = FastAPI(
  title="LangChain을 이용한 LLM 서비스",
  version = "1.0",
  description="Spin up a simple api server using Langchain's Runnable interfaces"
)

def rag_chain():
  dbname = "faiss_index_react"
  embeddings = HuggingFaceEmbeddings()

  vectordb = FAISS.load_local(
    folder_path=dbname, embeddings=embeddings, allow_dangerous_deserialization=True
  )

  llm = create_llm(project_id=project_id, credentials=credentials)
  retriever = vectordb.as_retriever()
  prompt_template = """\
  Use the provided context to answer the user's question. If you don't know the answer, say you don't know.

  Context:
  {context}

  Question:
  {question}"""

  rag_prompt = ChatPromptTemplate.from_template(prompt_template)

  entry_point_chain = RunnableParallel(
      {"context": retriever, "question": RunnablePassthrough()}
  )

  rag_chain = entry_point_chain | rag_prompt | llm | StrOutputParser()
  
  return rag_chain

@app.post("/upload")
async def upload_photo(file: UploadFile):
    UPLOAD_DIR = "./data"  # 이미지를 저장할 서버 경로
    
    content = await file.read()
    filename = file.filename  # uuid로 유니크한 파일명으로 변경
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

    upload_filename = UPLOAD_DIR + "/" + filename
    print("upload_filename:", upload_filename)
    reload(upload_filename)
    return {"filename": filename}

def reload(filename):
  split_docs = load_and_split(filename) 
  embeddings = HuggingFaceEmbeddings()
  dbname = "faiss_index_react"
  print("reload pdf to embedding...")
  embedding_vector(split_docs, embeddings, dbname)
  print("Completed embeddings")
  
@app.post("/rag")
def chain(query):
  dbname = "faiss_index_react"
  embeddings = HuggingFaceEmbeddings()

  vectordb = FAISS.load_local(
    folder_path=dbname, embeddings=embeddings, allow_dangerous_deserialization=True
  )

  llm = create_llm(project_id=project_id, credentials=credentials)
  retriever = vectordb.as_retriever()
  chain = create_stuff_docs_chain(llm, retriever)

  result = chain.invoke(query)
  
  return result
  

add_routes(
    app,
    rag_chain(),
    path="/watsonx-ai",
)