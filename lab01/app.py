import os
from dotenv import load_dotenv
from time import sleep
try:
    from langchain import PromptTemplate
    from langchain.chains import LLMChain, SimpleSequentialChain
    from langchain.document_loaders import PyPDFLoader
    from langchain.indexes import VectorstoreIndexCreator # Vectorize db index with chromadb
    from langchain.embeddings import HuggingFaceEmbeddings # For using HuggingFace embedding models
    from langchain.text_splitter import CharacterTextSplitter # Text splitter

    from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watson_machine_learning.foundation_models import Model
    from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
except ImportError as e:
    print(e)

load_dotenv()

print("Done importing dependencies.")

credentials = {
  "apikey": os.getenv("API_KEY"),
  "url": "https://us-south.ml.cloud.ibm.com"
}
project_id = os.getenv("PROJECT_ID", None)


print(credentials)
print("project_id:", project_id)

# Initialize the WatsonX model
params = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.TEMPERATURE: 0.2,
    GenParams.TOP_P: 1,
    GenParams.TOP_K: 25,
    GenParams.REPETITION_PENALTY: 1.0,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 20
}

llm_model = Model(
    model_id="google/flan-ul2",
    params=params,
    credentials=credentials,
    project_id=project_id
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
  
# Prompt Templates & Chains
prompt = PromptTemplate(
  input_variables = ["country"],
  template = "what is the capitial of {country}"
)

try:
  # In order to use Langchain, we need to instantiate Langchain extension
  lc_llm_model = WatsonxLLM(model = llm_model)
  
  chain = LLMChain(llm=lc_llm_model, prompt=prompt)
  
  countries = ["London", "Mexico", "Korea"]
  
  for country in countries:
    response = chain.run(country)
    print(prompt.format(country=country) + " = " + response.capitalize())
    sleep(0.5)
except Exception as e:
  print(e)
  
# 3. Simple sequential chains  
# Create two sequential prompts 
pt1 = PromptTemplate(
  input_variables = ["topic"],
  template = "Generate a random question about {topic}: Question: "
)

pt2 = PromptTemplate(
  input_variables = ["question"],
  template = "Answer the follow question: {question}"
)
print("done")

model_1 = Model(
  model_id = "google/flan-ul2",
  params = params,
  credentials = credentials,
  project_id = project_id
).to_langchain()

model_2 = Model(
  model_id = "google/flan-ul2",
  params = params,
  credentials = credentials,
  project_id = project_id
).to_langchain()

prompt_to_model_1 = LLMChain(llm=model_1, prompt=pt1)
prompt_to_model_2 = LLMChain(llm=model_2, prompt=pt2)

qa = SimpleSequentialChain(chains = [prompt_to_model_1, prompt_to_model_2], verbose=True)

try:
  qa.invoke("an animal")
except Exception as e:
  print(e)

# ---------------------------------------------------------------------------
# 4. Easy Loading of Documents Using Lang Chain
# LangChain을 사용하면 문서에서 구절을 쉽게 추출하여 문서 내용을 기반으로 질문에 답할 수 있습니다.
# ---------------------------------------------------------------------------

pdf = "./what-is-generative-ai.pdf"
loaders = [PyPDFLoader(pdf)]

# Index loaded PDF
index = VectorstoreIndexCreator(
  embedding = HuggingFaceEmbeddings(),
  text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=0)
).from_loaders(loaders)

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
    model_id="google/flan-ul2",
    params=params,
    credentials=credentials,
    project_id=project_id
).to_langchain()

# Init RAG chain
from langchain.chains import RetrievalQA 

chain = RetrievalQA.from_chain_type(
  llm=model,
  chain_type="stuff",
  retriever = index.vectorstore.as_retriever(),
  input_key = "question"
  )

# Answer based on the document
res = chain.invoke("what is Machine Learning?")
print(res)