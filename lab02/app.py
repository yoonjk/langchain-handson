import os 
from dotenv import load_dotenv 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

# langchain
# from langchain.document_loaders import PyPDFLoader 
from langchain import PromptTemplate 
from langchain_community.document_loaders import PyPDFLoader
# from langchain.vectorstores import Chroma  # deprecated
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings # deprecated
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv() 

credentials = {
  "apikey": os.getenv("API_KEY", None),
  "url" : os.getenv("API_URL", None)
}

project_id = os.getenv("PROJECT_ID", None)

params = {
  GenParams.DECODING_METHOD : "sample",
  GenParams.TEMPERATURE : 0.2,
  GenParams.TOP_P: 1,
  GenParams.TOP_K: 25,
  GenParams.MAX_NEW_TOKENS: 20,
  GenParams.MIN_NEW_TOKENS: 1,
  GenParams.REPETITION_PENALTY: 1.0,
}

llm_model = Model(
  model_id = "google/flan-ul2",
  params = params,
  credentials = credentials,
  project_id = project_id
)

print("Done initializing LLM.")

# Predict with the model
countries = ["France", "Japan", "Australia"]

try:
  for country in countries:
    question = f"What is the capital of {country}"
    res = llm_model.generate_text(question)
    print(f"The capital of {country} is {res.capitalize()}")
except Exception as e:
  print(e) 
  
# Initialize watsonx google/flan-ul2 model
params = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.TEMPERATURE: 0.2,
    GenParams.TOP_P: 1,
    GenParams.TOP_K: 100,
    GenParams.MIN_NEW_TOKENS: 50,
    GenParams.MAX_NEW_TOKENS: 300
}
model = Model(
    model_id=ModelTypes.FLAN_T5_XXL,
    params=params,
    credentials=credentials,
    project_id=project_id
).to_langchain()  


loader = PyPDFLoader("./data/LangChain.pdf")

# OpenAI의 Text Embedding 을 이용하여 chunk의 문서들을 임베딩 벡터로 변환하고, 
# Chroma Vector DB에 저장하고 인덱싱합니다. 
# 이때 메모리에만 남겨두는 것이 아니라 directory에 영구저장(persist)하여 추후 재사용할 수 있도록 합니다. 

text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=200)
pages = loader.load_and_split(text_splitter)

print("page_content:", pages[0].page_content)
print("meta:", pages[0].metadata)

vectordb = Chroma.from_documents(
  documents=pages, # Documents
  embedding=HuggingFaceEmbeddings(),
  persist_directory='db'
)

retriever = vectordb.as_retriever(
  search_type = "similarity", 
  search_kwagrs = {"k": 3}
)

llm_model = Model(
  model_id = 'meta-llama/llama-3-70b-instruct',
  params = params,
  credentials = credentials,
  project_id = project_id
).to_langchain()

# 검색QA(질문 답변 체인)를 구축하여 RAG 작업을 자동화하세요.
chain = RetrievalQA.from_chain_type(
  llm = llm_model,
  chain_type = "stuff",
  retriever = retriever,
  return_source_documents=True
)

# Answer based on the document
res = chain.invoke("langchain에 대해 한글로 설명해줘")

print(res["result"])

loader = PyPDFLoader("./data/Chain-of-Thought-prompting.pdf")
pages_new = loader.load_and_split(text_splitter)
_ = vectordb.add_documents(pages_new)

res = chain.invoke("what is chain-of-thought prompting?")
print(res["result"])

memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
  llm=llm_model, 
  retriever=vectordb.as_retriever(), 
  memory=memory)

query = "What is the langchain about?"
result = qa({"question": query})
print(result)

# RAG chain 생성
from langchain.schema.runnable import RunnablePassthrough
# pipe operator를 활용한 체인 생성
prompt = PromptTemplate(
  input_variables = ["question", "context"], 
  template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm_model 
)

res = rag_chain.invoke(query)
print(result)

loader = PyPDFLoader("./data/showers.pdf")
documents_new = loader.load_and_split(text_splitter)
_ = vectordb.add_documents(documents_new)

res = chain.invoke("이 소설의 제목은 뭐야?")

print(res)