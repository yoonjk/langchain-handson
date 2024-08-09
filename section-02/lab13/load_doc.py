
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.chains import RetrievalQA

def load_and_split(file_name: str):
  loader = PyPDFLoader(file_name)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size = 1000,
    chunk_overlap = 100
  )

  split_docs = text_splitter.split_documents(documents=documents)
  
  return split_docs

def embedding_vector(docs, embedding, dbname):
  """embedding docs to vectordb"""
  vectordb = FAISS.from_documents(documents=docs, embedding=embedding)
  vectordb.save_local(dbname)


def load_db(dbname, embeddings):
  vectordb = FAISS.load_local(
    folder_path=dbname, 
    embeddings=embeddings,
    allow_dangerous_deserialization=True
  )
  
  return vectordb

def create_stuff_docs_chain(llm, retriever):

  qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever = retriever
  )
  
  return qa