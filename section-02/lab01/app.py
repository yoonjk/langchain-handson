import os
from dotenv import load_dotenv

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.indexes import VectorstoreIndexCreator # Vectorize db index with chromadb
    from langchain_huggingface import HuggingFaceEmbeddings # For using HuggingFace embedding models
    from langchain.text_splitter import CharacterTextSplitter # Text splitter

    from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watson_machine_learning.foundation_models import Model
    from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
    # Init RAG chain
    from langchain.chains import RetrievalQA 
except ImportError as e:
    print(e)

load_dotenv()

print("Done importing dependencies.")

credentials = {
  "apikey": os.getenv("API_KEY"),
  "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)
# ---------------------------------------------------------------------------
# 4. Easy Loading of Documents Using Lang Chain
# LangChain을 사용하면 문서에서 구절을 쉽게 추출하여 문서 내용을 기반으로 질문에 답할 수 있습니다.
# ---------------------------------------------------------------------------

## Step 1 : Loading document
pdf = "./data/what-is-generative-ai.pdf"
loaders = [PyPDFLoader(pdf)]

# Step 2 : Split document
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)

# Step 3 : Embedding documents to memory
# Index loaded PDF
vectordb = VectorstoreIndexCreator(
  embedding = HuggingFaceEmbeddings(),
  text_splitter = text_splitter
).from_loaders(loaders)

# Step 4 : Define the LLM Model
# Initialize watsonx google/flan-ul2 model
params = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.TEMPERATURE: 0.2,
    GenParams.TOP_P: 1,
    GenParams.TOP_K: 100,
    GenParams.MIN_NEW_TOKENS: 50,
    GenParams.MAX_NEW_TOKENS: 300
}

llm_model = Model(
    model_id="google/flan-ul2",
    params=params,
    credentials=credentials,
    project_id=project_id
).to_langchain()



# Step 5 : Retrieval
chain = RetrievalQA.from_chain_type(
  llm=llm_model,
  chain_type="stuff",
  retriever = vectordb.vectorstore.as_retriever(),
  input_key = "question"
  )

# Answer based on the document
res = chain.invoke("what is Machine Learning?")
print(res)