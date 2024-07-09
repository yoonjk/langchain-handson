import os
from dotenv import load_dotenv 
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_ibm import WatsonxEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("API_KEY", None)
}

project_id = os.getenv("PROJECT_ID", None)

print(credentials)

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = loader.load_and_split(text_splitter)
embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
    )
vectordb = Chroma.from_documents(
	documents=documents, 
  embedding=embeddings
)



# Step 5. Set up a retriever
# 벡터 저장소를 리트리버로 설정하겠습니다. 
# 벡터 저장소에서 검색된 정보는 제너레이티브 모델에서 사용할 수 있는 추가 컨텍스트 또는 지식으로 사용됩니다.

retriever = vectordb.as_retriever()


# Step 6. Generate a response with a generative model


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

llm_model = Model(
    model_id="meta-llama/llama-3-70b-instruct",
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

llm = WatsonxLLM(
   model = llm_model
)

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in documents])
  
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke("what is the langchain?")  

print(res)
