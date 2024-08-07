
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

import os 
from dotenv import load_dotenv 

load_dotenv() 

credentials = {
  "apikey": os.getenv("API_KEY"),
  "url" : "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)


restaurant_prompt = """
신규 레스토랑의 네이밍 컨설팅을 하고 싶어요.
레스토랑 이름 목록을 제출하세요. 각 이름은 짧고 눈에 띄며 기억하기 쉬워야 합니다. 
레스토랑에 {restaurant_description} 좋은 이름은 무엇인가요? 
"""

params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.TEMPERATURE: 0.2
}
model_id = "meta-llama/llama-3-405b-instruct"
# model_id = "mistralai/mistral-large"


# Initialize the Watsonx foundation model
llm = WatsonxLLM(
    model_id=model_id,
    apikey=credentials['apikey'],
    url=credentials['url'],
    project_id=project_id,
    params=params
)

prompt= """
장미 꽃의 색은 
result : 빨간색
입력 : {flower} 꽃의 색을 형식으로 알려줘.
"""

prompt_template = PromptTemplate.from_template(template=prompt)


llm_chain = prompt_template | llm

result = llm_chain.invoke( {"flower" : '해바라기'})

print(result)

description = "신선한 양고기 수블라키와 다른 그리스 음식을 제공하는 "
description_02 = "야구 기념품을 테마로 한 버거 전문점"
description_03 = "라이브 하드록 음악과 기념품이 있는 카페 "

prompt_template = PromptTemplate.from_template(restaurant_prompt)

llm_chain = prompt_template| llm

result = llm_chain.invoke( {"restaurant_description" : description_02})

print(result)

print()
result = llm_chain.invoke( {"restaurant_description" : description_03})

print(result)