import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=temperature,
    )