from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


from langchain_ibm import WatsonxLLM

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

from dotenv import load_dotenv 
import os 

load_dotenv()

credentials = {
    "apikey": os.getenv("API_KEY", None),
    "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)

# PDF 파일 로드
loader = PyPDFLoader("data/showers.pdf")
document = loader.load()

print(document[0].page_content[:200])

# 스플리터 지정
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=3000,
    chunk_overlap=500
)

split_docs = text_splitter.split_documents(documents=document)

# 모델 정의 


params = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.TEMPERATURE: 2,
    GenParams.TOP_P: 0,
    GenParams.TOP_K: 100,
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.REPETITION_PENALTY: 1.2
}

llm = WatsonxLLM(
    model_id = "meta-llama/llama-3-405b-instruct",
    apikey = credentials['apikey'],
    url = credentials['url'],
    params = params, 
    project_id = project_id
)

# 총 분할된 도큐먼트 수
print(len(split_docs))

# 분할된 각 문서에 대한 요약 실행
# Map 단계에서 처리할 프롬프트 정의
# 분할된 문서에 적용할 프롬프트 내용을 기입합니다.
# 여기서 {pages} 변수에는 분할된 문서가 차례대로 대입되니다.
map_prompt= """다음은 문서 중 일부 내용입니다
{text}
이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
답변:"""

# Map 프롬프트 완성
map_template= PromptTemplate(template=map_prompt, input_variables=["text"])

# Reduce 단계에서 처리할 프롬프트 정의
reduce_prompt = """다음은 요약의 집합입니다:
{text}
이것들을 바탕으로 통합된 요약을 만들어 주세요.
답변:"""

# Reduce 프롬프트 완성
reduce_template = PromptTemplate(template=reduce_prompt , input_variables=["text"])

# 요약결과 출력
summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_template,
                                     combine_prompt=reduce_template,
                                     verbose=False
                                    )

LLM_response = summary_chain.invoke(split_docs)
print("LLM Reponse:", LLM_response['output_text'])

