
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM


def create_params(params: dict = None):
    if params:
        return params
     # 2. LLM 을 정의합니다
    params = {
		GenParams.DECODING_METHOD : "greedy", 
		GenParams.MIN_NEW_TOKENS: 1,
		GenParams.MAX_NEW_TOKENS: 500,
		GenParams.STOP_SEQUENCES : ['\n\n']
	}
 
    return params
    

def create_llm(credentials, project_id, model_name: str = 'meta-llama/llama-3-70b-instruct'):
    # 2. LLM 을 정의합니다
	params = {
		GenParams.DECODING_METHOD : "greedy", 
		GenParams.MIN_NEW_TOKENS: 1,
		GenParams.MAX_NEW_TOKENS: 500,
		GenParams.STOP_SEQUENCES : ['\n\n']
	}
    
	llm = WatsonxLLM(
		model_id = model_name,
		apikey = credentials['apikey'],
		url = credentials['url'],
		params = params, 
		project_id = project_id
	)
 
	return llm 

