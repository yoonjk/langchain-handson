from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from load_env import (
    credentials,
    project_id
)

from utils import (
    print_dash
)


from llm_model import create_llm

print_dash(125, '-')

# Step 1: Document Loading- Let’s use WikipediaLoader as the Document Loader.
loader = WikipediaLoader(query = 'Elon Musk', load_max_docs=5)
documents = loader.load()
print_dash(125, '-')
print('load')



# Step 2: Text Splitting- We will use RecursiveCharacterTextSplitter as the text splitter.
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 100)
docs = text_splitter.split_documents(documents=documents)
print_dash(125, '-')
print('split_documents')
print('doc len:', len(docs))

# Step 3: Embedding Function- For the embedding function, we will use HuggingFace BGE embedding model and let’s create a query.


model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device':'cpu'}

encode_kwargs = {'normalize_embeddings':True}

embedding_function = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

query = "Who is elon musk's father?"

# FAISS
"""
Let’s look into the functionalities of FAISS vector database, which makes use of the Facebook AI Similarity Search (FAISS) library.
"""


db = FAISS.from_documents(
    docs,
    embedding_function
)

# Similarity search with query
matched_docs = db.similarity_search(query = query, k = 1)
print_dash(125, '-')
print("similarity_search")
print(matched_docs)

# Similarity search with query vector
embedding_vector = embedding_function.embed_query(query)
matched_docs = db.similarity_search_by_vector(embedding_vector)
print_dash(125, '-')
print("similarity_search_by_vector")
print(matched_docs)

""" 
임베딩 함수는 둘 다 동일하므로 결과는 동일합니다.

대부분의 경우 반드시 인메모리 데이터베이스를 사용하지는 않습니다. 다른 오픈소스 데이터베이스를 만들어 보겠습니다.

# chroma
ChromaDB는 인메모리 데이터베이스와 영구 저장 옵션이 있는 백엔드로 모두 사용할 수 있습니다.
"""


db = Chroma.from_documents(docs, embedding_function, persist_directory="output/elon_muskdb")

loaded_db = Chroma(persist_directory = "output/elon_muskdb", embedding_function = embedding_function)

"""
마찬가지로 여기에서도 벡터를 사용한 유사도 검색뿐만 아니라 유사도 검색을 수행할 수 있습니다.
"""

# Similarity search with query
matched_docs = loaded_db.similarity_search(query = query, k = 5)
print_dash(125, '-')
print("similarity_search")
print(matched_docs)

# Similarity search with query vector
embedding_vector = embedding_function.embed_query(query)
matched_docs = loaded_db.similarity_search_by_vector(embedding_vector)
print_dash(125, '-')
print("similarity_search_by_vector")
print(matched_docs)

# Vector Store-backed retriever
"""
벡터 저장소를 사용하여 문서를 검색합니다. 기존에 보유하고 있는 ChromaDB 벡터 저장소를 사용하여 검색기를 구축해 보겠습니다.
"""
retriever = db.as_retriever()

query = "Who is elon musk's father?"
# get_relevant_documents deprecated
matched_docs = retriever.get_relevant_documents(query = query)

print_dash(125, '-')
print("similarity_search")
print(matched_docs)


"""
검색기를 만들 때 검색기가 문서를 검색하는 방법(최대 한계 관련성 검색-mmr 또는 유사도 검색 등)과 검색할 문서 수를 언급할 수도 있습니다.
"""

# Using MMR and limiting the number of retrieved documents to 1

retriever = db.as_retriever(search_type='mmr', search_kwargs={"k": 1})
matched_docs = retriever.get_relevant_documents(query=query)
print_dash(125, '-')
print("search_type='mmr' get_relevant_documents")
print(matched_docs)

# Using Similarity Search.
# Also keeping a minimum similarity threshold of 0.5 and retrieved documents = 2
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k" : 1})

print_dash(125, '-')
print("search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.5, 'k' : 2}, get_relevant_documents")
matched_docs = retriever.get_relevant_documents(query=query)
print(matched_docs)

# BM25 Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
print_dash(125, '-')
print("BM25 Retriever get_relevant_documents")
matched_docs = bm25_retriever.invoke(query)
print(matched_docs)