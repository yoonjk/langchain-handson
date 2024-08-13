from load_env import (
  credentials,
  project_id
)

from llm_model import create_llm
from rag import (
  format_docs,
  create_embedding,
  load_vector,
  load_and_splitter,
  create_splitter,
  embeddings_retriever,
  create_embedding,
  create_inmemorystore,
  create_multi_vector_retriever,
  generate_doc_ids,
  splitter_sub_docs
)
from chain import (
  create_chain,
  create_prompt_template
)

# 2. Define Model  
llm = create_llm(credentials=credentials, project_id=project_id)

# 3. Load and embedding
docs = load_and_splitter("./data/정책자료2022-05-02.pdf", chunk_size=20000, chunk_overlap=500)
embeddings = create_embedding() 
vector = load_vector(docs, embeddings=embeddings)
store = create_inmemorystore()
id_key = "doc_id"
retriever = create_multi_vector_retriever(vector=vector,store=store, id_key=id_key)
doc_ids = generate_doc_ids(docs)
child_text_splitter = create_splitter(chunk_size=1000, chunk_overlap=100)
sub_docs = splitter_sub_docs(id_key=id_key, docs = docs, doc_ids = doc_ids, child_text_splitter = child_text_splitter)
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
result = retriever.vectorstore.similarity_search("저 출산 원인이 무엇인가요?")[0]

print('child_page_content:', result)

result = retriever.invoke("저 출산 원인이 무엇인가요?")
print('------------------------')
print('page_content:', result)
