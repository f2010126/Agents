# Actual Agent
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool

from app.config import logger
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


def build_agent():
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0
        )

        tools = [
            Tool(
                name="search_codebase",
                func=search_codebase,
                description="Searchs codebase for relevant files and code snippets"
            )
        ]

        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT
        )

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True  # Allow to view
        )

        logger.info("Agent initialized successfully")
        return executor

    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        raise
