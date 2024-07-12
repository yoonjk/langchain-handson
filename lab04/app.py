

import os
from dotenv import load_dotenv 
import requests
# ibm_watson_machine_learning
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

# langchain
from langchain_ibm import WatsonxEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory


# langchain_community
from langchain_community.embeddings import TensorflowHubEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.utils import enforce_stop_tokens

load_dotenv()

def getBearer(apikey):
    form = {'apikey': apikey, 'grant_type': "urn:ibm:params:oauth:grant-type:apikey"}
    print("About to create bearer")
#    print(form)
    response = requests.post("https://iam.cloud.ibm.com/oidc/token", data = form)
    if response.status_code != 200:
        print("Bad response code retrieving token")
        raise Exception("Failed to get token, invalid status")
    json = response.json()
    if not json:
        print("Invalid/no JSON retrieving token")
        raise Exception("Failed to get token, invalid response")
    print("Bearer retrieved")
    
    return json.get("access_token")

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("API_KEY", None)
}

project_id = os.getenv("PROJECT_ID", None)

print([model.name for model in ModelTypes])

# Step 1: 문서 로드
loader = PyPDFLoader("./data/LangChain.pdf")

# Step 2: 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = loader.load_and_split(text_splitter)
text_chunks=[content.page_content for content in documents]

# Step 3 : 문서 벡터화(Embedding) with TensorflowHubEmbeddings
url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
embeddings  = TensorflowHubEmbeddings(model_url=url)

vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)



# Step 4: LLM 모델 정의

# Initialize the Watsonx foundation model
model_id = ModelTypes.LLAMA_2_70B_CHAT

parameters = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.TEMPERATURE: 2,
    GenParams.TOP_P: 0,
    GenParams.TOP_K: 100,
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.REPETITION_PENALTY:1.2,
    GenParams.STOP_SEQUENCES:['B)','\n'],
    GenParams.RETURN_OPTIONS: {'input_tokens': True,'generated_tokens': True, 'token_logprobs': True, 'token_ranks': True, }
}

llama_model = Model(
    model_id=model_id, 
    params=parameters, 
    credentials=credentials,
    project_id=project_id)

# to_langchain
llm=llama_model.to_langchain()

# Step 5: 관련 문서와 질문을 LLM에 던져
# 대화 내역에 대한 기억 (Chat History Memory) + Retrieval' 을 동시에 고려하여 사용자 질의에 대한 답변을 생성
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=vectorstore.as_retriever(), 
    memory=memory)

print("1. question==================================")
query = "what is the langchain?"
result = qa({"question": query})

print("1.1 result ==================================")
print(result)
print("1.2 result ==================================")
print(result["answer"])
print("2. question ==================================")

result = qa.invoke(query)
print("2.1. result ==================================")
print(result)
print("2.2. result ==================================")
print(result['answer'])