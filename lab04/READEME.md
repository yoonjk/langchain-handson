![](img/01-langchain.png)

## WatsonX with LangChain
안녕하세요, 여러분, 오늘은 LangChain에서 왓슨X.ai를 실행해 보겠습니다. 텍스트 문서를 읽고 질문을 해보겠습니다.

우리는 왓슨X.ai의 기초 모델을 사용할 것이며, 문서에서 벡터스토어를 생성하고 왓슨X와 함께 LangChain을 사용하는 방법을 보여드리겠습니다.

왓슨X를 랭체인과 연결하는 방법을 살펴볼 것입니다. 텍스트 문서를 읽고 이 문서에 질문을 할 것입니다.

소개  
LangChain은 오픈소스의 특성과 사용자 친화적인 접근 방식 덕분에 언어 모델(LLM) 애플리케이션 개발의 사실상의 표준으로 빠르게 자리 잡고 있습니다. 이 인기 있는 프레임워크는 프롬프트 템플릿, 출력 구문 분석, LLM 호출 시퀀싱, 세션 상태 유지 관리, RAG 사용 사례의 체계적 실행 등 다양한 LLM 관련 작업의 구현을 간소화합니다.
 
LangChain은 새로운 LLM 기능을 도입하지는 않지만, 파이썬과 자바스크립트에서 잘 구조화된 LLM 애플리케이션을 구축하기 위한 보완적인 프레임워크 역할을 합니다.
 

LangChain의 주요 기능 중 하나는 다양한 공급업체에서 개발 및 호스팅하는 LLM과의 호환성을 보장하는 공급업체 및 LLM 중립 API입니다. 예를 들어, LangChain에서 지원하는 여러 LLM 유형에 대한 체인(모델과 프롬프트로 구성)을 생성하는 것은 매우 간단합니다:

```python
chain = LLMChain(llm=llm, prompt=prompt).
```

왓슨엑스아이는 현재 다음과 같은 여러 랭체인 API를 통합한 WML(왓슨 머신러닝) API를 통해 랭체인에 대한 지원을 확장했습니다.
 
- LLMChain: 프롬프트와 LLM의 조합으로, 언어 모델 작업을 위한 기본 구조를 제공합니다.
- SimpleSequentialChain: 단계당 단일 입력/출력이 있는 선형 프로세스로, 한 단계의 출력이 다음 단계의 입력으로 사용됩니다.
- SequentialChain: 단계당 여러 개의 입력/출력을 허용하는 순차적 프로세스의 고급 버전입니다.
- TransformChain: 체인 내에 커스텀 transform() 함수를 통합하며, 일반적으로 LLM 입력/출력 변경에 사용됩니다.
- ConversationBufferMemory: 대화에서 상호작용의 기록을 유지하기 위해 이전 프롬프트와 응답을 저장합니다.

먼저 초기 라이브러리를 로드합니다.

```python
import os
from dotenv import load_dotenv
from typing import Any, List, Mapping, Optional, Union, Dict
from pydantic import BaseModel, Extra
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
```

```python
load_dotenv()
project_id = os.getenv("PROJECT_ID", None)
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("API_KEY", None)
}
```

왓슨X용 IBM Cloud에는 일반적으로 서비스를 배포할 수 있는 여러 리전이 있습니다. 일반적인 리전은 다음과 같습니다:

1. US South (Dallas) - us-south
2. US East (Washington, DC) - us-east
3. Europe (Frankfurt) - eu-de
4. Europe (London) - eu-gb
5. Asia Pacific (Tokyo) - jp-tok
6. Asia Pacific (Sydney) - au-syd

특정 지역의 왓슨 서비스 URL을 확인하려면 `https://REGION_ID.ml.cloud.ibm.com` 패턴을 따를 수 있습니다.

예를 들어, 해당 지역이 미국 남부(댈러스)인 경우 URL은 `https://us-south.ml.cloud.ibm.com`입니다.

Foundation Models on watsonx.ai

사용 가능한 모든 모델은 ModelTypes 클래스 아래에 표시됩니다. 자세한 내용은 문서를 참조하세요. 

```python
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
print([model.name for model in ModelTypes])
```

모델을 정의합니다.

```python
model_id = ModelTypes.LLAMA_2_70B_CHAT
```

## 모델 매개변수 정의하기
다른 모델이나 작업에 따라 모델 매개변수를 조정해야 할 수 있으며, 이를 위해서는 설명서를 참조하세요.

```python
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
```

```python
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 200
}
```

```python
#this cell should never fail, and will produce no output
import requests

def getBearer(apikey):
    form = {'apikey': apikey, 'grant_type': "urn:ibm:params:oauth:grant-type:apikey"}
    print("About to create bearer")
#    print(form)
    response = requests.post("https://iam.cloud.ibm.com/oidc/token", data = form)
    if response.status_code != 200:
        print("Bad response code retrieving token")
        raise Exception("Failed to get token, invalid status")
    json = response.json()
    if not json:
        print("Invalid/no JSON retrieving token")
        raise Exception("Failed to get token, invalid response")
    print("Bearer retrieved")
    return json.get("access_token")

credentials["token"] = getBearer(credentials["apikey"])
```

왓슨x.ai에서 라마 모델을 정의합니다.

```python
from ibm_watson_machine_learning.foundation_models import Model
# Initialize the Watsonx foundation model
llama_model = Model(
    model_id=model_id, 
    params=parameters, 
    credentials=credentials,
    project_id=project_id)
```

### 모델 세부 정보
로드된 모델에 대한 세부 정보를 보려면 다음을 입력하면 됩니다.

```python
llama_model.get_details()['short_description']
```

단순히 한계를 알고

```python
llama_model.get_details()['model_limits']
```

### Generation AI by using Llama-2-70b-chat model.
```python
instruction = "Using the directions below, answer in a maximum of  2 sentences. "
question = "What is the capital of Italy"
prompt=" ".join([instruction, question])
```

### Generate_text method
```python
llama_model.generate_text(question)
```

### Generate method
```python
result=llama_model.generate(prompt)['results'][0]['generated_text']
```

![](img/02-integration-langchain.png)

```python
from langchain import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
```

먼저 파운데이션 모델(예: 플랜 UL2)을 로드합니다.
```python
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 200
}

from ibm_watson_machine_learning.foundation_models import Model
# Initialize the Watsonx foundation model
flan_ul2_model = Model(
    model_id=ModelTypes.FLAN_UL2, 
    credentials=credentials,
    project_id=project_id,
    params=parameters
    
    )

prompt_template = "What color is the {flower}?"    
llm_chain = LLMChain(llm=flan_ul2_model.to_langchain(), 
                     prompt=PromptTemplate.from_template(prompt_template))


llm_chain('sunflower')

# {'flower': 'sunflower', 'text': 'yellow'}
```

### Remembering chat history

ConversationalRetrievalQA 체인은 RetrievalQAChain을 기반으로 구축되어 채팅 기록 구성 요소를 제공합니다.

먼저 채팅 기록(명시적으로 전달되거나 제공된 메모리에서 검색된)과 질문을 독립형 질문으로 결합한 다음 리트리버에서 관련 문서를 조회하고 마지막으로 해당 문서와 질문을 질문-답변 체인으로 전달하여 응답을 반환합니다.

리트리버를 만들려면 리트리버가 필요합니다. 아래 예에서는 임베딩으로 만들 수 있는 벡터 스토어에서 리트리버를 만들겠습니다.

arxiv에서 텍스트 문서를 읽어오겠습니다.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import TextLoader
loader = TextLoader("example.txt")
document = loader.load()
```

텍스로더로 문서를 로드할 때 이 객체의 데이터 유형은 텍스트의 일부 목록인 langchain.schema.document.Document입니다.

### Splitting the document
```python
text_splitter = CharacterTextSplitter(separator="\n",
                                      chunk_size=1000, 
                                      chunk_overlap=200)
# Split text into chunks
documents = text_splitter.split_documents(document)

type(documents[0].page_content)

documents[0].page_content
```

### Embeddings
텍스트를 임베드하는 방법에는 여러 가지가 있지만, 이 데모에서는 텐서플로우 임베딩을 사용하겠습니다.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import TensorflowHubEmbeddings

url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"


embeddings  = TensorflowHubEmbeddings(model_url=url)
text_chunks=[content.page_content for content in documents]

vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
```

이제 입력/출력을 추적하고 대화를 유지하는 데 필요한 메모리 객체를 만들 수 있습니다.
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)
```

이제 ConversationalRetrievalChain 체인을 초기화합니다.

```python
#llm=flan_ul2_model.to_langchain()
llm=llama_model.to_langchain()
qa = ConversationalRetrievalChain.from_llm(llm=llm, 
                                           retriever=vectorstore.as_retriever(), 
                                           memory=memory)

query = "What is the topic about"
result = qa({"question": query})
result["answer"]

# result
# The topic is about the Transformer model in deep learning, specifically discussing its use of self-attention and multi-head attention mechanisms.

```

LangChain의 메모리 구현은 프롬프트에 입력과 출력을 모두 추가하여 LLM의 stateless 특성을 해결합니다. 이 접근 방식은 이해하고 구현하기 쉽지만, 호스팅된 인스턴스에 대한 LLM 토큰 제한과 토큰 비용을 고려해야 합니다.

이러한 문제를 완화하기 위해 LangChain은 상호작용과 토큰 한도를 관리하기 위해 서로 다른 전략을 사용하는 

- ConversationBufferWindowMemory  
- ConversationSummaryBufferMemory  
- ConversationTokenBufferMemory  

와 같은 다양한 메모리 유형을 제공하고 있으며, 이러한 메모리 유형은 상호 작용과 토큰 한도를 관리합니다. 

### Pass in chat history
위의 예에서는 메모리 객체를 사용하여 채팅 기록을 추적했습니다. 명시적으로 전달할 수도 있습니다. 이렇게 하려면 메모리 객체 없이 체인을 초기화해야 합니다.
```python
qa = ConversationalRetrievalChain.from_llm(llm=llama_model.to_langchain(),
                                           retriever=vectorstore.as_retriever())

```

다음은 왓슨x와의 채팅 기록이 없는 상태에서 질문하는 예시입니다.
```python
chat_history = []
query = "What is the topic  about"
result = qa({"question": query, "chat_history": chat_history})

result["answer"]

' The topic is about the Transformer model in natural language processing, specifically discussing the use of self-attention and multi-head attention in the model.'

```

다음은 채팅 기록이 있는 질문의 예입니다.

```python
chat_history = [(query, result["answer"])]
query = "What is Transformer"
result = qa({"question": query, "chat_history": chat_history})

result['answer']
```

결과

```python
'\nThe Transformer model is a sequence transduction model that relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions. It is based on attention mechanisms and is used for various tasks such as machine translation, English constituency parsing, and text summarization. The Transformer model is the first transduction model that uses self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolutions. It has been shown to be superior in quality, more parallelizable, and requiring significantly less time to train than other sequence transduction models.'
```

채팅 기록

```python
chat_history

[('What is the topic  about',
  ' The topic is about the Transformer model in natural language processing, specifically discussing the use of self-attention and multi-head attention in the model.')]
```
  


