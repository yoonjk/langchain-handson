from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods, ModelTypes
from ibm_watson_machine_learning.foundation_models.model import Model 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from dotenv import load_dotenv 
import os

load_dotenv()

credentials = {
  "apikey" : os.getenv("API_KEY", None),
  "url" : "https://us-south.ml.cloud.ibm.com"
}

print(credentials)

project_id = os.getenv("PROJECT_ID", None)

prompt = ChatPromptTemplate(
  input_variables=["content"],
  messages = [
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

params = {
  GenParams.DECODING_METHOD : DecodingMethods.GREEDY,
  GenParams.MIN_NEW_TOKENS: 1,
  GenParams.MAX_NEW_TOKENS: 100
}

model = Model(
  model_id = ModelTypes.LLAMA_2_70B_CHAT.value, 
  credentials = credentials, 
  params = params,
  project_id = project_id
)


llm = WatsonxLLM(model = model)

chain = LLMChain(llm = llm, prompt = prompt)

while True: 
  content = input(">> ")
  result = chain({"content": content})
  print(result)
  