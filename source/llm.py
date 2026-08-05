import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


class MedicalLLM:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def generate_answer(self, question, context):

        prompt = f"""
You are a medical assistant.

Answer ONLY from the provided context.

If the answer is not found, reply:

"I couldn't find enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        return response.content.strip()