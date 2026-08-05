from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    """
    Create a FAISS vector store from document chunks.
    """

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print("✅ FAISS vector store created.")

    return vector_store


def save_vector_store(vector_store, save_path="vector_store"):
    """
    Save FAISS index locally.
    """

    vector_store.save_local(save_path)

    print(f"✅ Vector store saved to '{save_path}'.")