from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(name="search_agent",
                   model="gemini-2.5-flash-lite",
                   description="Uses google search to answer stuff",
                   instruction="answer based on the facts and the search. You are an expert and you do not halucinate",
                   tools=[google_search])
