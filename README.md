# DataMind - An AI that thinks over data, both structured (SQL) and unstructured (RAG).


# 🤖 DataMind — Unified AI Assistant for SQL, RAG & Hybrid Intelligence

> *DataMind* is an intelligent Streamlit-based assistant that connects directly to your databases and combines structured (SQL) and unstructured (RAG) reasoning.  
> It can automatically decide whether your question requires SQL queries, text retrieval, or both — powered by **Google Gemini** (with future **Ollama** support).

---

## 🚀 Features

### 🧾 SQL Agent
- Natural language to SQL conversion using LangChain.
- Executes real-time database queries.
- Automatic error handling and friendly messages.

### 📚 RAG (Retrieval-Augmented Generation)
- Fetches knowledge or text from Postgres tables.
- Builds semantic vector indexes using `SentenceTransformer` + FAISS.
- Answers contextual, descriptive, or document-related queries.

### 🧠 Hybrid & Auto Modes
- Combines both SQL + RAG reasoning for complex queries.
- **Auto Mode:** The AI decides the best approach automatically (SQL, RAG, or Hybrid).
- Smart fallback — if query parsing fails, it explains logically.

### 💡 Intelligent Routing
- Uses Gemini to analyze queries and route them dynamically.
- Falls back to traditional keyword detection when unsure.

### ⚙️ Easy Integration
- Works with PostgreSQL out of the box.
- Gemini API integration via `.env`.
- Ollama local model support reserved for future versions.

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **LLM** | Gemini (Google Generative AI) |
| **DB Layer** | PostgreSQL via psycopg2 |
| **Vector Store** | FAISS + SentenceTransformer |
| **Agent Framework** | LangChain |
| **Env Management** | python-dotenv |

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/DevShah-DevsTreeIT/DataMind.git
cd DataMind
