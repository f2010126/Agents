from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    temperature=0.7
)

# -------------------------
# NEW MEMORY STORE (replaces ConversationBufferMemory)
# -------------------------
store = {}


def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# -------------------------
# Prompt (same idea, but IMPORTANT: placeholder is still correct)
# -------------------------
conversation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use conversation history."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])


# Create a chain with memory
base_chain = conversation_prompt | llm

# -------------------------
# ATTACH MEMORY (THIS WAS MISSING)
# -------------------------
conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Helper function to handle conversation


def chat(message):
    response = conversation_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": "user_1"}}
    )
    return response.content


# Have a conversation
print(chat("Hi, my name is Alice"))
print(chat("What's my name?"))
print(chat("Tell me a joke about my name"))
