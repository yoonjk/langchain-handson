from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
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

memory = ConversationBufferMemory(
  chat_memory = FileChatMessageHistory("messages.json"),
  memory_key="messages", return_messages=True
)
prompt = ChatPromptTemplate(
  input_variables=["content", "messages"],
  messages = [
    MessagesPlaceholder(variable_name = "messages"),
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

params = {
  GenParams.DECODING_METHOD : DecodingMethods.GREEDY,
  GenParams.MIN_NEW_TOKENS: 1,
  GenParams.MAX_NEW_TOKENS: 30
}

model = Model(
  model_id = ModelTypes.FLAN_T5_XXL.value,
  credentials = credentials, 
  params = params,
  project_id = project_id
)


llm = WatsonxLLM(model = model)

chain = LLMChain(llm = llm, prompt = prompt, memory = memory)

while True: 
  content = input(">> ")
  result = chain({"content": content})
  print(result)
  