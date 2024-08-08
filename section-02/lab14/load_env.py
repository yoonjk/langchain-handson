import os 
from dotenv import load_dotenv 

load_dotenv() 

# 1. 도구를 정의합니다 ##########
# Tavily search engine
tavily_api_key= os.getenv("TAVILY_API_KEY", None)
os.environ['TAVILY_API_KEY'] = tavily_api_key


credentials = {
	"apikey": os.getenv("API_KEY", None),
  "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)

