
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM

def create_llm(credentials, project_id):
    # 2. LLM 을 정의합니다
	params = {
		GenParams.DECODING_METHOD : "greedy", 
		GenParams.MIN_NEW_TOKENS: 1,
		GenParams.MAX_NEW_TOKENS: 500,
		GenParams.TEMPERATURE: 0.2,
		GenParams.STOP_SEQUENCES : ['\n\n']
	}
 
	llm = WatsonxLLM(
		model_id = 'meta-llama/llama-3-70b-instruct',
		apikey = credentials['apikey'],
		url = credentials['url'],
		params = params, 
		project_id = project_id
	)
 
	return llm 

