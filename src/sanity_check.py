# src/sanity_check.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = model.invoke("Say 'setup working' and nothing else.")
print(response.content)