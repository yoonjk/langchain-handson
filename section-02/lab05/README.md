
![](img/01-rag-overview-using-chroma.png)

Use watsonx Granite Model Series, Chroma, and LangChain to answer questions (RAG)

Disclaimers
- 왓슨엑스 컨텍스트에서 사용할 수 있는 프로젝트와 스페이스만 사용하세요.

**About Retrieval Augmented Generation**
검색 증강 생성(RAG)은 자연어로 지식 베이스를 쿼리하는 등 사실에 입각한 정보 리콜이 필요한 다양한 사용 사례에 활용할 수 있는 다용도 패턴입니다.

가장 간단한 형태의 RAG는 3단계가 필요합니다:

- 지식창고 구절 색인화(한 번)  
- 지식창고에서 관련 구절을 검색합니다(모든 사용자 쿼리에 대해).  
- 검색된 구절을 대규모 언어 모델에 공급하여 응답을 생성합니다(모든 사용자 쿼리에 대해).  


**Contents**
This notebook contains the following parts:

- [Set up the environment](#set-up-the-environment)
- [Document data loading](#document-data-loading)
- [Build up knowledge base](#build-up-knowledge-base)
- [Foundation Models on watsonx.ai](#foundation-models-on-watsonxai)
- [Generate a retrieval-augmented response to a question](#generate-a-retrieval-augmented-response-to-a-question)




#### Set up the environment
이 노트북의 샘플 코드를 사용하기 전에 다음 설정 작업을 수행해야 합니다:  
- 왓슨 머신 러닝(WML) 서비스 인스턴스를 만듭니다(무료 플랜이 제공되며 인스턴스 생성 방법에 대한 정보는 여기에서 확인할 수 있습니다).

Install and import the dependecies
```python
!pip install "langchain==0.1.10" | tail -n 1
!pip install "ibm-watsonx-ai>=0.2.6" | tail -n 1
!pip install -U langchain_ibm | tail -n 1
!pip install wget | tail -n 1
!pip install sentence-transformers | tail -n 1
!pip install "chromadb==0.3.26" | tail -n 1
!pip install "pydantic==1.10.0" | tail -n 1
!pip install "sqlalchemy==2.0.1" | tail -n 1
```

```python
import os, getpass
```

**watsonx API connection**
이 셀은 기초 모델 추론을 위해 watsonx API로 작업하는 데 필요한 자격 증명을 정의합니다.  

Action:  IBM 클라우드 사용자 API 키를 입력합니다. 자세한 내용은 [문서]([documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui))를 참조하세요..

```python
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": getpass.getpass("Please enter your WML api key (hit enter): ")
}
```

**Defining the project id**
API를 사용하려면 호출의 컨텍스트를 제공하는 프로젝트 ID가 필요합니다. 이 노트북이 실행 중인 프로젝트에서 ID를 가져옵니다. 그렇지 않은 경우, 프로젝트 ID를 입력해 주세요.

**Hint**: project_id는 다음과 같이 찾을 수 있습니다. watsonx.ai에서 프롬프트 랩을 엽니다. UI 맨 위에는 프로젝트 / <프로젝트 이름> /이 있습니다. <프로젝트 이름> 링크를 클릭합니다. 그런 다음 프로젝트의 관리 탭(프로젝트 -> 관리 -> 일반 -> 세부 정보)에서 project_id를 가져옵니다.

```python
try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = input("Please enter your project_id (hit enter): ")
```

#### Document data loading
State of the Union으로 파일 다운로드
```python
import wget

filename = 'state_of_the_union.txt'
url = 'https://raw.github.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'

if not os.path.isfile(filename):
    wget.download(url, out=filename)
```

#### Build up knowledge base
RAG에서 가장 일반적인 접근 방식은 주어진 사용자 쿼리에 대한 의미적 유사성을 계산하기 위해 지식창고의 고밀도 벡터 표현을 만드는 것입니다.  

이 기본 예제에서는 연두교서 연설 콘텐츠(파일 이름)를 가져와서 청크로 분할하고 오픈 소스 임베딩 모델을 사용하여 임베딩한 다음 [Chroma](https://www.trychroma.com/)에 로드한 다음 쿼리합니다

```python
from langchain.document_loaders import TextLoader 
from langchain.text_spiltter import CharacterTextSplitter
from langchain.vectorstores import Chroma

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```

우리가 사용하고 있는 데이터 세트는 이미 chroma 에서 수집할 수 있는 독립된 구절로 분할되어 있습니다.

**Create an embedding function**
사용자 정의 임베딩 함수를 공급하여 chromadb에서 사용할 수 있습니다. 사용하는 임베딩 모델에 따라 크로마 데이터베이스의 성능이 달라질 수 있습니다. 다음 예제에서는 watsonx.ai 임베딩 서비스를 사용합니다. 사용 가능한 임베딩 모델은 get_embedding_model_specs를 사용하여 확인할 수 있습니다.  

```python
from langchain_ibm import WatsonxEmbeddings 
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes 

embeddings = WatsonxEmbeddings(
  model_id = EmbeddingTypes.IBM_SLATE_30M_ENG.value,
  credentials = credentials,
  project_id = project_id
)

docsearch = Chroma.from_documents(texts, embeddings)
```

#### Foundation Models on watsonx.ai
IBM 왓슨x 파운데이션 모델은 Langchain에서 지원하는 LLM 모델 목록 중 하나입니다. 이 예시는 Langchain을 사용하여 Granite 모델 시리즈와 통신하는 방법을 보여줍니다.

**Defining model**
추론에 사용할 model_id를 지정해야 합니다:

```python
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}
```

**LangChain CustomLLM wrapper for watsonx model**
정의된 파라미터와 ibm/granite-13b-chat-v2를 사용하여 Langchain에서 WatsonxLLM 클래스를 초기화합니다.  

```python
from langchain_ibm import WatsonxLLM

llm_watsonx = WatsonxLLM(
    model_id=model_id.value,
    credentials = credentials,
    project_id=project_id,
    params=parameters
)
```

#### Generate a retrieval-augmented response to a question
검색QA(질문 답변 체인)를 구축하여 RAG 작업을 자동화하세요.
```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
  llm=llm_watsonx, 
  chain_type="stuff", 
  retriever=docsearch.as_retriever())
```

**Select questions**
Get questions from the previously loaded test dataset.
```python
query = "What did the president say about Ketanji Brown Jackson"
qa.invoke(query)
```
