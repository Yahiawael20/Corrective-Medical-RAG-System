from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    Load the HuggingFace embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("✅ Embedding model loaded.")

    return embeddings