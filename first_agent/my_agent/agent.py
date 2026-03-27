from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search, AgentTool

research_agent = Agent(
    name="Researcher",
    model='gemini-2.5-flash-lite',
    instruction="""You are a specialized research agent. Your only job is to use the  
    google_search tool to find top 5 news for a given topic. Do not answer any user questions directly.""",
    tools=[google_search],
    output_key="research_result"

)
print("Resereach Agent created successfully.")

summariser = Agent(
    name="Summariser",
    model='gemini-2.5-flash-lite',
    instruction="""  
    Read the research findings {research_result} and create a summary for each topic with the link to read more  
    """,
    output_key="summary_result"
)


# Root Coordinator
root_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction="""  
    You are coordinator. First your job is to delegate user questions to the 'research_agent' to gather information. Second pass the findings to the 'summarizert' agent to create a summary. Finally, compile the summaries into a final response for the user.  
    """,
    tools=[AgentTool(research_agent),
           AgentTool(summariser),
           ]
)
