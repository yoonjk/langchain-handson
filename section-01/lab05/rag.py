from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid 

def load_and_splitter(pdf_file_path, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Loader document"""
    loader = PyPDFLoader(pdf_file_path)
  
    text_splitter = create_splitter(splitter_name="RECURSIVE", chunk_overlap = chunk_overlap, chunk_size = chunk_size)
    splitter_docs = loader.load_and_split(text_splitter=text_splitter)

    return splitter_docs

def create_splitter(splitter_name : str = "RECURSIVE", chunk_size: int = 1000, chunk_overlap: int = 200):
    """create text_splitter"""
    print('chunk_size:', chunk_size)
    print('chunk_overlap:', chunk_overlap)
    if splitter_name:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n"])
    else:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return text_splitter

def load_vector(docs, embeddings):
    """vectore store embedding"""
    vector = FAISS.from_documents(documents=docs, embedding = embeddings)
  
    return vector

def embeddings_retriever(docs, embeddings):
    """get vector"""
    vector = create_embedding(docs=docs, embeddings=embeddings)
    
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
    
def create_inmemorystore():
    """create InMemoryByteStore"""
    store = InMemoryByteStore()
    
    return store

def create_multi_vector_retriever(id_key, vector, store):
    """create multi vector retriever"""
    retriever = MultiVectorRetriever(
        vectorstore=vector,
        byte_store=store,
        id_key = id_key,
        search_kwargs={"k": 1}
    )
    
    return retriever 

def generate_doc_ids(docs):
    """docFor multi vector  """
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    
    print('doc_ids:', len(doc_ids))
    
    return doc_ids

def splitter_sub_docs(id_key, docs, doc_ids, child_text_splitter):
    sub_docs = []
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for sub_doc in _sub_docs:
            sub_doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)
        
    return sub_docs
            

