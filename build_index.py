from source.loaders import load_documents
from source.splitter import split_documents
from source.embeddings import get_embeddings
from source.vector_store import create_vector_store, save_vector_store


def main():

    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Loading embedding model...")
    embeddings = get_embeddings()

    print("Creating FAISS index...")
    vector_store = create_vector_store(chunks, embeddings)

    print("Saving vector store...")
    save_vector_store(vector_store)

    print("\n🎉 Index built successfully!")


if __name__ == "__main__":
    main()