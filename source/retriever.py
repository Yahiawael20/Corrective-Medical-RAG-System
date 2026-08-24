from langchain_community.vectorstores import FAISS
from source.embeddings import get_embeddings


class Retriever:

    def __init__(self, vector_path="vector_store"):

        embeddings = get_embeddings()

        self.vector_store = FAISS.load_local(
            folder_path=vector_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅ Vector Store Loaded.")

    def retrieve(self, query, k=5):

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )

        docs = []
        scores = []

        for doc, score in results:
            docs.append(doc)
            scores.append(score)

        return docs, scores