import os 
from dotenv import load_dotenv 

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain  
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.model import Model 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 
load_dotenv()

credentials = {
  "apikey" : os.getenv("API_KEY", None),
  "url" : "https://us-south.ml.cloud.ibm.com"
}

print(credentials)

project_id = os.getenv("PROJECT_ID", None)

[print(model.value)  for model in ModelTypes]

model_id = ModelTypes.LLAMA_2_70B_CHAT

print("model_id:", model_id)

params = {
  GenParams.DECODING_METHOD : DecodingMethods.GREEDY,
  GenParams.MAX_NEW_TOKENS : 100, 
  GenParams.MIN_NEW_TOKENS : 1
}

model = Model(
  model_id = model_id.value,
  credentials = credentials,
  params = params,
  project_id = project_id
)

code_prompt = PromptTemplate(
  template = "write a very short {language} function that will {task}",
  input_variables = ["language", "task"]
)

test_prompt = PromptTemplate(
  template = "Write a test for the following {language} code:\n{code}",
  input_variables = ["language", "code"]
)

code_chain = LLMChain(llm = model.to_langchain(), 
                      prompt = code_prompt,
                      output_key = "code")
test_chain = LLMChain(llm = model.to_langchain(), 
                      prompt = test_prompt,
                      output_key = "test")


chain = SequentialChain(
  chains = [code_chain, test_chain],
  input_variables = ["task", "language"],
  output_variables = ["test", "code"]
)

result = chain.invoke({"language": 'python', "task": 'return a list of numbers'})

print(result)