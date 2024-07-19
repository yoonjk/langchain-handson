
import streamlit as st
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from langchain.prompts.chat import ChatPromptTemplate
from langserve import RemoteRunnable

st.title("watsonx.ai를 이용한 챗봇!")

def langserve_api(prompts, langserv_url):
    llm = RemoteRunnable(langserv_url)
    prompt_template = "{question}"
    prompt = ChatPromptTemplate.from_template(
	    prompt_template
    )
    prompt_payload = {"question": prompts }
    
    result = llm.invoke(prompt_payload)
    print('result:', result)

    return result

with st.sidebar:
    langserv_url = st.text_input('LangServe Url:', value="http://localhost:8000/watsonx-ai")

    if not (langserv_url):
        st.warning('Please enter Url', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
 
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = langserve_api(prompt, langserv_url) 
            st.write(response) 
    st.session_state.messages.append({"role": "assistant", "content": response})