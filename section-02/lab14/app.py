from langchain_community.document_loaders import YoutubeLoader

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from load_env import (
  credentials,
  project_id
)
from llm_model import create_llm

# 1. Simple Videos
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=QsYGlZkevEg", add_video_info=True)

docs = loader.load()

print (type(docs))
print (f"Found video from {docs[0].metadata['author']} that is {docs[0].metadata['length']} seconds long")
print ("")
print (docs)

llm = create_llm(credentials=credentials, project_id=project_id)

chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
result = chain.run(docs)
print('stuff:', result)

# Reduce
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts[:4])

# 3. Multiple Videos
youtube_url_list = ["https://www.youtube.com/watch?v=AXq0QHUwmh8", "https://www.youtube.com/watch?v=EwHrjZxAT7g"]

texts = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

for url in youtube_url_list:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    result = loader.load()
    
    texts.extend(text_splitter.split_documents(result))
    
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
result = chain.run(texts)

print("====================")
print('multi loader:', result)


