from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a simple template
template = """You are a helpful assistant that translates {input_language} to {output_language}.

Text: {text}

Translation:"""

# Create the prompt template
prompt = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template=template
)

# Use the template
formatted_prompt = prompt.format(
    input_language="English",
    output_language="Spanish",
    text="Hello, how are you?"
)

print(formatted_prompt)
print("#### USING IT IN AN LLM ####")

# Output will be:
# You are a helpful assistant that translates English to Spanish.
#
# Text: Hello, how are you?
#
# Translation:


# Initialize the LLM with API key
# Make sure you have GOOGLE_API_KEY in your .env file
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chain = prompt | model

response = chain.invoke({
    "input_language": "English",
    "output_language": "Spanish",
    "text": "Hello, how are you?"
})

print(response.content)
