from langchain_community.tools.tavily_search import TavilySearchResults 
from langchain.agents import AgentExecutor,create_json_chat_agent
from langchain_fireworks import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as ui

ui.title("Social Media Agent")

wt = ui.text_input('Enter the topic you')
but = ui.button("Submit")
if but:
    if(len(wt)>0):
        tavily_tool = TavilySearchResults(max_results=1)
        tools = [tavily_tool]
        llm =  ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct",api_key="xnhQeAxEwERoqGcTqJs3YoRs294xZAoBjj1fnp6Pg5T1z11w")


        human = '''TOOLS
        ------
        Assistant can ask the user to use tools to look up information that may be helpful in  answering the users original question. The tools the human can use are:

        {tools}

        RESPONSE FORMAT INSTRUCTIONS
        ----------------------------

        When responding to me, please output a response in one of two formats:

        **Option 1:**
        Use this if you want the human to use a tool.
        Markdown code snippet formatted in the following schema:

        ```json
        {{
            "action": string, \ The action to take. Must be one of {tool_names}
            "action_input": string \ The input to the action
            "url": string
        }}
        ```

        **Option #2:**
        Use this if you want to respond directly to the human. Markdown code snippet formatted             in the following schema:

        ```json
        {{
            "action": "Final Answer",
            "action_input": string \ You should put what you want to return to and add url 
            "url": string
        }}
        ```
        USER'S INPUT
        --------------------
        Here is the user's input (remember to respond with a markdown code snippet of a json             blob with a single action, url used and NOTHING else):

        {input} '''

        prompt = ChatPromptTemplate.from_messages(
            [   ("human", human),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        
        
        agent = create_json_chat_agent(llm,tools,prompt)
        agent_exectutor = AgentExecutor(agent=agent,tools=tools,verbose=True,handle_parsing_errors=True)
        ques = wt.strip()
        ans = agent_exectutor.invoke({"input":ques})
        ui.write(f"Your Query: {ques}")
        ui.write(f"What I have found for your query: {ans['output']}")

