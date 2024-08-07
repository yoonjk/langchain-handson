from fastapi import FastAPI 
from langchain.prompts import ChatPromptTemplate 
from langchain.chains import LLMChain 
from langserve import add_routes
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams 

# user define module
from chain import chain

app = FastAPI(
	title = "LangChain Server with watsonx.ai",
 	version = "1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain(),
    path="/watsonx-ai",
)

