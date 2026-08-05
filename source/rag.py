from source.query_rewriter import QueryRewriter
from source.retriever import Retriever
from source.evaluator import ContextEvaluator
from source.llm import MedicalLLM


class CorrectiveRAG:

    def __init__(self):

        self.rewriter = QueryRewriter()
        self.retriever = Retriever()
        self.evaluator = ContextEvaluator()
        self.llm = MedicalLLM()

    def ask(self, question):

        # ==========================================
        # Step 1: Rewrite Question
        # ==========================================
        rewritten_question = self.rewriter.rewrite(question)

        # ==========================================
        # Step 2: Retrieve Documents
        # ==========================================
        docs, scores = self.retriever.retrieve(
            rewritten_question,
            k=5
        )

        # ==========================================
        # Step 3: Evaluate Retrieval
        # ==========================================
        evaluation = self.evaluator.evaluate(scores)

        quality = evaluation["status"]
        avg_score = evaluation["average_score"]

        # ==========================================
        # Step 4: Corrective Retrieval
        # ==========================================
        if quality == "Weak":

            rewritten_question = self.rewriter.rewrite(
                rewritten_question
            )

            docs, scores = self.retriever.retrieve(
                rewritten_question,
                k=5
            )

            evaluation = self.evaluator.evaluate(scores)

            quality = evaluation["status"]
            avg_score = evaluation["average_score"]

        # ==========================================
        # Step 5: Build Context
        # ==========================================
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # ==========================================
        # Step 6: Generate Answer
        # ==========================================
        answer = self.llm.generate_answer(
            question=question,
            context=context
        )

        # ==========================================
        # Step 7: Return Results
        # ==========================================
        return {
            "original_question": question,
            "rewritten_question": rewritten_question,
            "answer": answer,
            "quality": quality,
            "average_score": avg_score,
            "documents": docs
        }