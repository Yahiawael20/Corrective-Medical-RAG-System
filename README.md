# 🏥 Corrective Medical RAG System

A Corrective Retrieval-Augmented Generation (Corrective RAG) system that answers medical questions using trusted medical PDF documents.

The system retrieves the most relevant medical passages, evaluates the retrieval quality, automatically rewrites unclear questions when necessary, and finally generates an accurate answer grounded in the retrieved context.

---

## 🚀 Features

- 📄 Load multiple medical PDF documents
- ✂️ Intelligent text chunking
- 🔎 FAISS Vector Database
- 🤖 Sentence Transformers Embeddings
- 📊 Retrieval Quality Evaluation
- ✍️ Automatic Query Rewriting
- 🧠 LLM-based Answer Generation
- 📚 Source Citation with Page Numbers
- 🌐 Interactive Streamlit Interface

---

## 📂 Dataset

The knowledge base contains medical documents covering:

- Tuberculosis
- Hypertension
- Diabetes
- Malaria
- Nutrition & Malnutrition
- Mental Health

---

## 🏗️ Project Structure

```
Corrective_RAG/
│
├── data/
│   ├── *.pdf
│
├── source/
│   ├── loaders.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── evaluator.py
│   ├── query_rewriter.py
│   ├── answer_generator.py
│   └── rag.py
│
├── vector_store/
│
├── app.py
├── build_index.py
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Pipeline

```
User Question
      │
      ▼
Query Rewriter
      │
      ▼
Retriever (FAISS)
      │
      ▼
Context Evaluator
      │
      ├───────────────► Strong Context
      │                     │
      │                     ▼
      │              Answer Generator
      │
      └───────────────► Weak Context
                            │
                            ▼
                  Rewrite Query Again
                            │
                            ▼
                        Retrieve Again
                            │
                            ▼
                    Answer Generator
```

---

## 🛠️ Technologies

- Python
- LangChain
- FAISS
- Sentence Transformers
- HuggingFace Embeddings
- Groq LLM
- Streamlit
- dotenv

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Corrective-RAG.git
cd Corrective-RAG
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GROQ_API_KEY=your_api_key
```

---

## 📦 Build the Vector Database

```bash
python build_index.py
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 💬 Example Questions

- What are the symptoms of tuberculosis?
- How is hypertension treated?
- What causes malaria?
- What are the complications of diabetes?
- What is protein-energy malnutrition?
- What are the goals of the Mental Health Action Plan?

---

## 📊 Output

The system displays:

- Original Question
- Rewritten Question (if applicable)
- Context Quality (Strong / Weak / Poor)
- Similarity Score
- Final Answer
- Source Document
- Page Number

