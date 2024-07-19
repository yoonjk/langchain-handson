from langchain.prompts.chat import ChatPromptTemplate
from langserve import RemoteRunnable

llm = RemoteRunnable("http://localhost:8000/watsonx-ai")


prompt_template = "{question}"
prompt = ChatPromptTemplate.from_template(
	prompt_template
)

result = llm.invoke({"question":"나는 여의도에 있어요. 오늘 날씨 어때?"})
print('result:', result)

