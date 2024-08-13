from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings

def load_and_splitter(pdf_file_path):
    """ document loader """
    loader = PyPDFLoader(pdf_file_path)
  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n"])
    splitter_docs = loader.load_and_split(text_splitter=text_splitter)

    return splitter_docs

def vectordb(docs, embeddings):
    """vectore store embedding"""
    vector = FAISS.from_documents(documents=docs, embedding = embeddings)
  
    return vector

def embeddings_retriever(docs, embeddings):
    """get vector db"""
    vector = vectordb(docs=docs, embeddings=embeddings)
    
    return vector.as_retriever()

def format_docs(docs):
    """context documents"""
    format_docs = "\n\n".join([doc.page_content for doc in docs])
  
    return format_docs

def create_embedding(embedding_name : str = "huggingface"):
    """ get embedding"""
    
    if embedding_name == "huggingface":
        return HuggingFaceEmbeddings()
    else:
        raise Exception("Not found embedding")