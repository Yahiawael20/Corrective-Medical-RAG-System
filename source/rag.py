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

    def _needs_rewrite(self, question):
        """
        Decide whether the raw question is clear enough
        to search directly or whether it should be rewritten.

        The question is considered ambiguous if:
        - It contains fewer than 4 words
        - OR it does not contain a question mark
        """

        q = question.strip()

        return len(q.split()) < 4 or "?" not in q

    def ask(self, question):

        # ============================================================
        # Step 1: Rewrite Question (only if needed)
        # ============================================================

        query_rewritten = False

        if self._needs_rewrite(question):

            rewritten_question = self.rewriter.rewrite(question)

            query_rewritten = True

        else:

            rewritten_question = question

        # ============================================================
        # Step 2: Retrieve Documents
        # ============================================================

        docs, scores = self.retriever.retrieve(
            rewritten_question,
            k=5
        )

        # ============================================================
        # Step 3: Evaluate Retrieval
        # ============================================================

        evaluation = self.evaluator.evaluate(scores)

        quality = evaluation["status"]
        avg_score = evaluation["average_score"]

        # ============================================================
        # Step 4: Corrective Retrieval
        # ============================================================
        #
        # If retrieval quality is Weak or Poor:
        # - Rewrite the ORIGINAL question
        # - Retrieve again
        # - Evaluate again
        #
        # Maximum 2 corrective attempts
        # ============================================================

        max_attempts = 2
        attempt = 0

        while quality in ("Weak", "Poor") and attempt < max_attempts:

            attempt += 1

            # Rewrite from the ORIGINAL question
            # to avoid query drift
            rewritten_question = self.rewriter.rewrite(question)

            # A rewrite happened
            query_rewritten = True

            # Retrieve again
            docs, scores = self.retriever.retrieve(
                rewritten_question,
                k=5
            )

            # Evaluate the new retrieval
            evaluation = self.evaluator.evaluate(scores)

            quality = evaluation["status"]
            avg_score = evaluation["average_score"]

        # ============================================================
        # Step 5: Build Context
        # ============================================================

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # ============================================================
        # Step 6: Generate Answer
        # ============================================================

        answer = self.llm.generate_answer(
            question=question,
            context=context
        )

        # ============================================================
        # Step 7: Return Results
        # ============================================================

        return {
            "original_question": question,
            "rewritten_question": rewritten_question,
            "query_rewritten": query_rewritten,
            "answer": answer,
            "quality": quality,
            "average_score": avg_score,
            "documents": docs
        }