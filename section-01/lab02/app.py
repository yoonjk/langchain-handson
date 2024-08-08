
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

import os 
from dotenv import load_dotenv 
import json

load_dotenv() 

credentials = {
  "apikey": os.getenv("API_KEY"),
  "url" : "https://us-south.ml.cloud.ibm.com",
  "project_id": os.getenv("PROJECT_ID", None)
}


params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.STOP_SEQUENCES: ["\n\n"]

}
model_id = "meta-llama/llama-3-405b-instruct"


# Initialize the Watsonx foundation model
llm = WatsonxLLM(
    model_id=model_id,
    apikey=credentials['apikey'],
    url=credentials['url'],
    project_id=credentials['project_id'],
    params=params
)

prompt= """

question : {flower} 꽃의 색을 JSON 형식으로 보여줘
result : 
"""

prompt_template = PromptTemplate.from_template(template=prompt)


llm_chain = prompt_template | llm

result = llm_chain.invoke( {"flower" : '백합'})
print(result)
