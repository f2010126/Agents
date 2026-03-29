from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load Var
load_dotenv()


# init
llm = ChatGoogleGenerativeAI(
    # Use latest model: gemini-2.0-flash, gemini-1.5-pro, etc.
    model='gemini-2.5-flash-lite',
    temperature=0.7
)


prompt = ChatPromptTemplate.from_messages([
    # personality/instruction to the model. System message
    ("system", "You are a helpful assistant that explains concepts clearly."),
    ("human", "Explain {topic} in simple terms")
])
# chain
chain = prompt | llm

# Run the chain
# sends this input into the prompt and then thats inserted into the mode;
result = chain.invoke({"topic": "quantum computing"})
print(result.content)
