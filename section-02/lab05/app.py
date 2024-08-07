import os  
from dotenv import load_dotenv 
import wget 

# loading  and split documents
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader 
from langchain_community.vectorstores import Chroma 
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain.chains import RetrievalQA 

from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes 

# Define Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods 

load_dotenv() 

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("API_KEY", None)
}

project_id = os.getenv("PROJECT_ID", None)

filename = "state_of_the_union.txt"
url = "https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt"

if not os.path.isfile(filename):
  wget.download(url, out=filename)

# Step 1 Loading document
loader = TextLoader(filename)
documents = loader.load() 

# Step 2 : Spliter
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Step 3 : Embedding with watsonx
embeddings = WatsonxEmbeddings(
  model_id = EmbeddingTypes.IBM_SLATE_30M_ENG.value,
  url = credentials["url"],
  apikey = credentials['apikey'],
  project_id = project_id
)

vectordb = Chroma.from_documents(
  documents = texts, 
  embedding=embeddings)

# Define model

model_id = ModelTypes.GRANITE_13B_CHAT_V2
# Define the model parameters 

params = {
  GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
  GenParams.MIN_NEW_TOKENS: 1,
  GenParams.MAX_NEW_TOKENS: 100,
  GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

llm_model = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=params
)

# Generate a retrieval-augmented response to a question 

qa = RetrievalQA.from_chain_type(
  llm = llm_model,
  chain_type="stuff",
  retriever = vectordb.as_retriever()
)
query = "What did the president say about Ketanji Brown Jackson"
res = qa.invoke(query)
print(res['result'])