from langchain.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.render import render_text_description_and_args, render_text_description
from langchain_google_community import GoogleSearchAPIWrapper

from datetime import datetime 

# setup the tools
@tool 
def dummy():
  """Dummy function"""
  pass

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers"""

    return a / b 

@tool
def get_todays_date():
    """Get today's date in YYYY-MM-DD format."""
    global date
    date = datetime.now().strftime("%Y-%m-%d")
    return date

google_search = GoogleSearchAPIWrapper()


@tool
def top5_results(query):
    """ Get google search"""
    return google_search.results(query, 3)
  
# k=5은 검색 결과를 5개까지 가져오겠다는 의미입니다

create_tavily_search = TavilySearchResults()
  
def load_tools(extool):
	# Define tool
	# 1-3. tools 리스트에 도구 목록을 추가
  if extool:
    tools = [dummy, extool]
  else:  
    tools = [dummy]
  # tools.extend(load_tools(['wikipedia']))
  
  return tools 

def create_tools_info(prompt, tools):
  prompt = prompt.partial(
    tools=render_text_description_and_args(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
  )
  
  return prompt 

