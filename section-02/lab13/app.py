
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

embeddings = HuggingFaceEmbeddings()
llm = create_llm(project_id=project_id, credentials=credentials)
dbname = "faiss_index_react"
vectordb = load_db(dbname=dbname, embeddings=embeddings)
retriever = vectordb.as_retriever()
chain = create_stuff_docs_chain(llm, retriever)

query ="이 소설의 내용을 한글로 요약해줘." 
result = chain.invoke(query)

print(result)
