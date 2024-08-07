from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

from langchain_ibm import WatsonxLLM

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

import os 
from dotenv import load_dotenv

load_dotenv()

credentials = {
    "apikey": os.getenv("API_KEY", None),
    "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = os.getenv("PROJECT_ID", None)

loader = YoutubeLoader.from_youtube_url( "https://www.youtube.com/watch?v=txOv_pi-_R4", add_video_info=False)
list_of_doc_objects = loader.load()



print("**** Number of document objects ****")
print(str(len(list_of_doc_objects)))

print("**** Text of document ****")
text = list_of_doc_objects[0].page_content
print(text)

words = text.split()
word_count = len(words)
print ("Number of words = " + str(word_count))
char_count = len(text)
print ("Number of characters = " + str(char_count))
num_tokens =    int (char_count/4)  # Using a thumb rule of 4 chars per token
print ("Approx Number of tokens = " + str(num_tokens))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)  #8000 characters or 2000 tokens
chunks = text_splitter.split_documents(list_of_doc_objects)
print("Number of Chunks: " + str(len(chunks)))

for x in range (0, len(chunks)):
    print("Chunk Number: "+str(x) +" " + chunks[x].page_content + "\n")





#model parameters
parameters = {   
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}



model_id = ModelTypes.LLAMA_2_70B_CHAT  #LLM Model selected

llm = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)



map_prompt = "Write a concise summary of the following:'{text}' CONCISE SUMMARY: "
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

combine_prompt = "Write a concise abstractive summary of the following:'{text}' Summary should include financial numbers."
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                     verbose=False
                                    )

LLM_response = summary_chain.invoke(chunks)
print(LLM_response.get("output_text"))

map_prompt = "Write a concise summary of the following:'{text}' CONCISE SUMMARY: "
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

combine_prompt = "Write a concise bullet point summary of the following:'{text}'. The summary should include financial numbers."
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                     verbose=False
                                    )

LLM_response = summary_chain.invoke(chunks)
print(LLM_response.get("output_text"))