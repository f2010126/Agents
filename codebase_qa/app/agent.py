# Actual Agent
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool

from app.tools import search_codebase
from app.prompts import SYSTEM_PROMPT


def build_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0
    )

    tools = [
        Tool(
            name="search_codebase",
            func=search_codebase,
            description="Search the codebase for relevant files and code snippets"
        )
    ]

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )
