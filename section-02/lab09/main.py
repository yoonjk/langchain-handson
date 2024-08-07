import os 
from dotenv import load_dotenv 

from langchain_community.document_loaders import TextLoader 
from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
from langchain.text_splitter import CharacterTextSplitter 

from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.model import Model 
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 


load_dotenv() 

credentials = {
  "apikey": os.getenv("API_KEY", None),
  "url" : "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)

loader = TextLoader("./data/facts.txt")

text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size = 200, 
  chunk_overlap=0
)

documents = loader.load_and_split(
  text_splitter = text_splitter
)

embedding = HuggingFaceEmbeddings()

# exists document
db = Chroma(
  embedding_function=embedding,
  persist_directory="db"
)

# loading document
# db = Chroma.from_documents(
#   documents = documents,
#   embedding=embedding,
#   persist_directory="db"
# )

params = {
  GenParams.DECODING_METHOD : "greedy",
  GenParams.MIN_NEW_TOKENS : 1, 
  GenParams.MAX_NEW_TOKENS : 100
}

llm = WatsonxLLM(
  model_id =  ModelTypes.LLAMA_2_70B_CHAT.value,
  url = credentials["url"],
  apikey = credentials['apikey'],
  params = params,
  project_id = project_id
)

qa = RetrievalQA.from_chain_type(
  llm = llm, 
  chain_type = "stuff", 
  retriever = db.as_retriever(),
  return_source_documents = True
)

result = qa.invoke("What is an interesting fact about the english language?")
  
print(result['result'])
