import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

SUMMARY_PROMPT = """
You are an investment analyst. Given the annual report text, summarize the company's performance.
Provide a BUY / HOLD / SELL verdict with clear reasoning.

Report:
{report_text}
"""

def summarize_report(text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["report_text"],
        template=SUMMARY_PROMPT
    )
    return llm.predict(prompt.format(report_text=text[:7000]))  # safe context limit
