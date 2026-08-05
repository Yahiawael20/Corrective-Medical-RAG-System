import numpy as np


class ContextEvaluator:
    """
    Evaluate retrieved context quality based on FAISS distances.
    Lower distance = Better retrieval.
    """

    def evaluate(self, scores):

        if not scores:
            return {
                "status": "Poor",
                "best_score": None,
                "average_score": None
            }

        scores = [float(score) for score in scores]

        best_score = min(scores)
        average_score = np.mean(scores)

        # FAISS Distance Thresholds
        if best_score <= 0.60:
            status = "Strong"

        elif best_score <= 0.80:
            status = "Moderate"

        else:
            status = "Weak"

        return {
            "status": status,
            "best_score": round(best_score, 4),
            "average_score": round(float(average_score), 4)
        }