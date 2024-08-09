from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_prompt():
  
  system_prompt = """Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}
	Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
	Valid "action" values: "Final Answer" or {tool_names}
	Provide only ONE action per $JSON_BLOB, as shown:"
	```
	{{
		"action": $TOOL_NAME,
		"action_input": $INPUT
	}}
	```
	Follow this format:
	Question: input question to answer
	Thought: consider previous and subsequent steps
	Action:
	```
	$JSON_BLOB
	```
	Observation: action result
	... (repeat Thought/Action/Observation N times)
	Thought: I know what to respond
	Action:
	```
	{{
		"action": "Final Answer",
		"action_input": "Final response to human"
	}}
	Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
	Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""

  human_prompt = """{input}
	{agent_scratchpad}
	(reminder to always respond in a JSON blob)"""

  prompt = ChatPromptTemplate.from_messages(
			[
					("system", system_prompt),
					MessagesPlaceholder("chat_history", optional=True),
					("human", human_prompt),
			]
	)
  
  return prompt