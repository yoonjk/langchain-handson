from load_env import (
  credentials,
  project_id
)

from load_smith_env import (
  LANGCHAIN_API_KEY,
  LANGCHAIN_PROJECT,
  LANGCHAIN_ENDPOINT, 
  LANGCHAIN_TRACING_V2,
)
from llm_model import create_llm
from rag import (
  format_docs,
  load_and_splitter,
  embeddings_retriever,
  create_embedding,
)
from chain import (
  create_chain,
  create_prompt_template
)

# 2. Define Model  
llm = create_llm(credentials=credentials, project_id=project_id)

# 3. Load and embedding
docs = load_and_splitter("./data/정책자료 2022-05-02.pdf")
embeddings = create_embedding()
retriever = embeddings_retriever(docs = docs, embeddings=embeddings)

# 4. Define chain
prompt_template = """
{context}
question : {question}
"""
prompt = create_prompt_template(prompt_template)
chain = create_chain(llm = llm, retriever=retriever, prompt=prompt, format_docs=format_docs)

# 5. Query
result = chain.invoke("이책에 대해 요약해줘. You answer the korea language")

print(result)

