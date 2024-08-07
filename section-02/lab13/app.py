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

print(credentials)
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

print(params)

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
print(LLM_response.get("output_text"))

# 질문 템플릿 형식 정의
template = """다음은 소설에 대한 요약본입니다. 
다음의 내용을 독서 감상문 형식으로 작성해 주세요. 

독서 감상문의 형식은 다음과 같습니다:

처음: 글을 읽게 된 동기나 책을 처음 대했을 때의 느낌을 쓰고, 글의 종류나 지은이 소개, 주인
공이나 주제의 소개
중간: 주인공의 행동과 나의 행동을 비교해 보기도 하고, 글의 내용을 평가해 보기도 하며, 글
속에서 발견한 주제나 의미가 우리 사회에 어떻게 작용할 것인가를 씁니다. 그리고 글을 읽으면서 받은
감동을 쓰기도 합니다.
끝: 글의 내용을 정리하며, 교훈을 적어두기도 한다. 그리고 끝글은 지루하지 않도록 산뜻하게

{text}

답변:
"""

# 템플릿 완성
prompt = PromptTemplate(template=template, input_variables=['text'])

