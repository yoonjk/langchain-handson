import os, wget
from dotenv import load_dotenv


# langchain 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma 
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain.chains import RetrievalQA 
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# watsonx
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods, EmbeddingTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams 

load_dotenv() 

credentials = {
  "apikey": os.getenv("API_KEY", None),
  "url" : "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID")

filename = "data/showers.pdf"

# Load documents
load = PyPDFLoader(filename)
documents = load.load()
print('document[0]:', documents[0].page_content[:200])

# Split documents
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# embeddings
embeddings = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(
  documents = texts, 
  embedding=embeddings)


# Define Model
params = {
  GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
  GenParams.MIN_NEW_TOKENS: 1,
  GenParams.MAX_NEW_TOKENS: 1000,
  GenParams.STOP_SEQUENCES: ['<|endoftext|>']
}

llm_model = WatsonxLLM(
  model_id = "meta-llama/llama-3-70b-instruct",
  apikey = credentials['apikey'],
  url = credentials['url'],
  project_id = project_id,
  params = params
)

# Generate LLM
qa = RetrievalQA.from_chain_type(
  llm = llm_model,
  chain_type = "stuff",
  retriever = vectordb.as_retriever()
)

query = "이소셜의 제목은 뭐야?"
res = qa.invoke(query)
print(res['result'])

query = "이 소셜을 한글로 요약해줘"
res = qa.invoke(query)
print(res['result'])