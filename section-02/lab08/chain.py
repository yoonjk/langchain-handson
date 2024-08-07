from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams 
from langchain.prompts import ChatPromptTemplate 
from langchain.chains import LLMChain 
import os 
from dotenv import load_dotenv 

load_dotenv()

credentials = {
		"apikey": os.getenv("API_KEY", None),
		"url": "https://us-south.ml.cloud.ibm.com"
	}

project_id = os.getenv("PROJECT_ID", None)

params = {
			GenParams.DECODING_METHOD: "sample",
			GenParams.TEMPERATURE: 0.2,
			GenParams.TOP_P: 1,
			GenParams.TOP_K: 100,
			GenParams.MIN_NEW_TOKENS: 50,
			GenParams.MAX_NEW_TOKENS: 300
}

model = Model(
		model_id = "meta-llama/llama-3-70b-instruct",
		credentials=credentials,
		project_id = project_id,
		params = params
	)
 
 
def chain():

  prompt_template = "{question}"
  
  prompt = ChatPromptTemplate.from_template(
	  prompt_template
  )

  llm_chain = LLMChain(llm = model.to_langchain(), prompt=prompt)
  
  return prompt | llm_chain | parser

def parser(result):
  print(result)
  
  return result['text']

