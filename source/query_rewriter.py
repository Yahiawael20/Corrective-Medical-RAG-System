import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


load_dotenv()


class QueryRewriter:

    def __init__(self):

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def rewrite(self, question):

        prompt = f"""
You are a query rewriter for a medical Corrective RAG system.

Your job is to improve a user's question ONLY when it is necessary
for better document retrieval.

IMPORTANT RULES:

1. Do NOT rewrite every question.
2. If the question is already clear, specific, and suitable for
   document retrieval, return it EXACTLY unchanged.
3. If the question is short, vague, incomplete, ambiguous, or poorly
   phrased, rewrite it into a clear and specific medical question.
4. Preserve the original meaning.
5. Do NOT add medical information that is not implied by the question.
6. Do NOT answer the question.
7. Use medical terminology when appropriate.
8. Return ONLY the final question.
9. Do NOT add explanations, labels, quotes, or extra text.

Examples:

Input:
What are the symptoms of tuberculosis?

Output:
What are the symptoms of tuberculosis?

Input:
TB symptoms

Output:
What are the symptoms of tuberculosis?

Input:
causes of high BP

Output:
What are the causes of high blood pressure?

Input:
malnutrition causes

Output:
What are the major causes of malnutrition?

Input:
What is pulmonary tuberculosis?

Output:
What is pulmonary tuberculosis?

Input:
tell me about TB

Output:
What are the main clinical features and characteristics of tuberculosis?

Now process this question:

{question}
"""

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        rewritten = response.content.strip()

        return rewritten