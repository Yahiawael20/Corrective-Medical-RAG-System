import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


class QueryRewriter:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def rewrite(self, question):

        prompt = f"""
Rewrite the following medical question to improve document retrieval.

Rules:
- Keep the meaning.
- Use complete medical terminology.
- Do NOT answer.
- Return ONLY the rewritten question.

Question:
{question}
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        return response.content.strip()