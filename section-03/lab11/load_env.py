import os 
from dotenv import load_dotenv 

load_dotenv() 

# 1. 도구를 정의합니다 ##########
# Tavily search engine
tavily_api_key= os.getenv("TAVILY_API_KEY", None)
os.environ['TAVILY_API_KEY'] = tavily_api_key

# Google Search API
google_api_key = os.getenv("GOOGLE_API_KEY", None)
google_cse_id = os.getenv("GOOGLE_CSE_ID", None)

os.environ['GOOGLE_API_KEY'] = google_api_key
os.environ['GOOGLE_CSE_ID'] = google_cse_id

credentials = {
	"apikey": os.getenv("API_KEY", None),
  "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)

nasa_key = os.getenv("NASA_KEY", None)