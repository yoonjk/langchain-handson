from load_env import (
  credentials,
  project_id
)

from llm_model import create_llm
from load_doc import (
  load_and_split,
  embedding_vector
)
from langchain_huggingface import HuggingFaceEmbeddings

# docs = load_and_split("data/이효석-메밀꽃필무렵.pdf")
docs = load_and_split("data/김유정-동백꽃.pdf")
embeddings = HuggingFaceEmbeddings()
dbname = "faiss_index_react"
vectordb = embedding_vector(docs, embeddings, dbname)
