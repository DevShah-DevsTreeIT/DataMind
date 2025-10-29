import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------------------
# Setup
# -------------------------------------
load_dotenv()
st.set_page_config(page_title="Unified AI Assistant", layout="wide")
st.title("🤖 Unified AI Assistant — SQL + RAG + Hybrid + Auto")

# -------------------------------------
# Sidebar Config
# -------------------------------------
st.sidebar.header("⚙️ Settings")

# Database
db_type = st.sidebar.selectbox("Database", ["PostgreSQL"], index=0)
host = st.sidebar.text_input("Host", "localhost")
port = st.sidebar.text_input("Port", "5432")
dbname = st.sidebar.text_input("Database Name", "postgres")
user = st.sidebar.text_input("User", "postgres")
password = st.sidebar.text_input("Password", type="password")

# Model Selection
st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Select Model", ["Gemini (Google API)", "Ollama (Local)"], index=0)

# Mode Selection (Added Auto Mode)
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode",
    ["SQL Agent 🧾", "RAG 📚", "Hybrid 🧠", "Auto 🤖"],
    index=3
)

# -------------------------------------
# Connection & Model Setup
# -------------------------------------
def init_llm(choice):
    """Initialize Gemini or fallback."""
    if choice == "Gemini (Google API)":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found. Please add it to your environment.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Ollama temporarily disabled
    st.warning("⚠️ Ollama integration is currently disabled. Using Gemini instead.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found. Please add it to your environment.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if "llm" not in st.session_state:
    st.session_state.llm = init_llm(model_choice)

if st.sidebar.button("🔌 Connect Database"):
    try:
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        st.session_state.db = SQLDatabase.from_uri(conn_str)
        st.success("✅ Database Connected!")
    except Exception as e:
        st.error(f"Failed to connect: {e}")

# -------------------------------------
# SQL Agent Setup
# -------------------------------------
def get_sql_agent():
    if "sql_agent" not in st.session_state:
        try:
            st.session_state.sql_agent = create_sql_agent(
                llm=st.session_state.llm,
                db=st.session_state.db,
                verbose=True,
                handle_parsing_errors=True  # helps in catching output parsing issues
            )
        except Exception as e:
            st.error(f"SQL Agent Error: {e}")
    return st.session_state.sql_agent

# -------------------------------------
# RAG Setup
# -------------------------------------
def connect_pg():
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)

def fetch_texts(table, col):
    conn = connect_pg()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL")
    data = [r[col] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return data

def build_rag_index(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    return index, model

def query_rag(query):
    if "rag_index" not in st.session_state:
        st.warning("⚠️ No RAG index found. Please build one first.")
        return ""
    q_vec = st.session_state.rag_model.encode([query])
    D, I = st.session_state.rag_index.search(np.array(q_vec, dtype=np.float32), len(st.session_state.rag_texts))
    retrieved = [st.session_state.rag_texts[i] for i in I[0]]
    context = "\n---\n".join(retrieved)
    llm = init_llm(model_choice)
    prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else response

# -------------------------------------
# Query Detection Helpers
# -------------------------------------
def detect_query_type(query: str) -> str:
    """Simple heuristic for keyword detection."""
    sql_keywords = [
        "count", "average", "total", "sum", "list", "how many", "show", "table", "column",
        "group by", "where", "filter", "join"
    ]
    if any(kw in query.lower() for kw in sql_keywords):
        return "SQL"
    return "RAG"

def ai_decide_query_type(query: str) -> str:
    """Ask the LLM to decide whether to use SQL, RAG, or Hybrid."""
    llm = st.session_state.llm
    decision_prompt = f"""
    You are an intelligent router. Given the following user question, decide which mode should be used:
    - SQL: for database or structured queries (tables, counts, filters, etc.)
    - RAG: for general knowledge, document or context-based questions.
    - Hybrid: when both structured data and context might help.

    Only reply with ONE word: SQL, RAG, or Hybrid.

    User Question: "{query}"
    """
    try:
        decision = llm.invoke(decision_prompt)
        choice = decision.content.strip().upper()
        if choice not in ["SQL", "RAG", "HYBRID"]:
            choice = detect_query_type(query)
        return choice
    except Exception as e:
        st.warning(f"⚠️ Error during AI mode decision: {e}")
        return detect_query_type(query)

# -------------------------------------
# Error-safe Execution Wrapper
# -------------------------------------
def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ValueError as ve:
        match = re.search(r'["\'](.+?)["\']', str(ve))
        user_msg = match.group(1) if match else str(ve)
        st.warning(f"⚠️ The AI encountered a problem: {user_msg}")
    except Exception as e:
        st.warning(f"⚠️ Unexpected error: {str(e)}")

# -------------------------------------
# Interface per Mode
# -------------------------------------
if mode == "SQL Agent 🧾":
    st.subheader("🧾 SQL Agent Mode")
    query = st.text_area("Ask something about your data:")
    if st.button("Run Query"):
        agent = get_sql_agent()
        with st.spinner("Generating SQL and fetching results..."):
            result = safe_execute(agent.invoke, query)
            if result:
                st.write(result["output"])

elif mode == "RAG 📚":
    st.subheader("📚 RAG Mode")
    table = st.text_input("Table Name", "your_table")
    column = st.text_input("Text Column", "description")

    if st.button("📄 Build RAG Index"):
        with st.spinner("Building RAG Index..."):
            texts = fetch_texts(table, column)
            index, model = build_rag_index(texts)
            st.session_state.rag_index = index
            st.session_state.rag_model = model
            st.session_state.rag_texts = texts
            st.success(f"Indexed {len(texts)} text records!")

    st.divider()
    query = st.text_area("Ask a question:")
    if st.button("Ask (RAG)") and "rag_index" in st.session_state:
        with st.spinner("Thinking..."):
            answer = safe_execute(query_rag, query)
            if answer:
                st.write(answer)

elif mode == "Hybrid 🧠":
    st.subheader("🧠 Hybrid Smart Mode")
    query = st.text_area("Ask anything (the model decides best approach):")
    if st.button("Ask (Hybrid)"):
        with st.spinner("Analyzing your query..."):
            query_type = detect_query_type(query)
            st.info(f"🤖 Detected Query Type: {query_type}")

            if query_type == "SQL":
                agent = get_sql_agent()
                result = safe_execute(agent.invoke, query)
                if result:
                    st.success("💾 SQL Mode Result:")
                    st.write(result["output"])
            else:
                answer = safe_execute(query_rag, query)
                if answer:
                    st.success("📚 RAG Mode Result:")
                    st.write(answer)

elif mode == "Auto 🤖":
    st.subheader("🤖 Auto Smart Mode (LLM decides the best approach automatically)")
    query = st.text_area("Ask anything — AI will decide the best mode:")
    if st.button("Ask (Auto)"):
        with st.spinner("Analyzing your question..."):
            chosen_mode = ai_decide_query_type(query)
            st.info(f"🧭 AI decided to use: **{chosen_mode} Mode**")

            if chosen_mode == "SQL":
                agent = get_sql_agent()
                result = safe_execute(agent.invoke, query)
                if result:
                    st.success("💾 SQL Result:")
                    st.write(result["output"])

            elif chosen_mode == "RAG":
                answer = safe_execute(query_rag, query)
                if answer:
                    st.success("📚 RAG Result:")
                    st.write(answer)

            elif chosen_mode == "HYBRID":
                # Run both and combine
                agent = get_sql_agent()
                sql_res = safe_execute(agent.invoke, query)
                rag_res = safe_execute(query_rag, query)
                st.success("🧠 Hybrid Combined Result:")
                if sql_res:
                    st.write("💾 SQL Output:")
                    st.write(sql_res["output"])
                if rag_res:
                    st.write("📚 RAG Output:")
                    st.write(rag_res)

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Unified AI Assistant — SQL + RAG + Hybrid + Auto (Gemini powered)")






# import os
# import re
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import create_sql_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
# import psycopg2
# from psycopg2.extras import RealDictCursor
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss

# # -------------------------------------
# # Setup
# # -------------------------------------
# load_dotenv()
# st.set_page_config(page_title="Unified AI Assistant", layout="wide")
# st.title("🤖 Unified AI Assistant — SQL + RAG + Hybrid")

# # -------------------------------------
# # Sidebar Config
# # -------------------------------------
# st.sidebar.header("⚙️ Settings")

# # Database
# db_type = st.sidebar.selectbox("Database", ["PostgreSQL"], index=0)
# host = st.sidebar.text_input("Host", "localhost")
# port = st.sidebar.text_input("Port", "5432")
# dbname = st.sidebar.text_input("Database Name", "postgres")
# user = st.sidebar.text_input("User", "postgres")
# password = st.sidebar.text_input("Password", type="password")

# # Model Selection (Ollama visible, but inactive)
# st.sidebar.markdown("---")
# model_choice = st.sidebar.selectbox("Select Model", ["Gemini (Google API)", "Ollama (Local)"], index=0)

# # Mode Selection
# st.sidebar.markdown("---")
# mode = st.sidebar.radio(
#     "Mode",
#     ["SQL Agent 🧾", "RAG 📚", "Hybrid 🧠"],
#     index=2
# )

# # -------------------------------------
# # Connection & Model Setup
# # -------------------------------------
# def init_llm(choice):
#     """Initialize only Gemini for now."""
#     if choice == "Gemini (Google API)":
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             st.error("❌ GOOGLE_API_KEY not found. Please add it to your environment.")
#             st.stop()
#         return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
#     # Ollama temporarily disabled
#     st.warning("⚠️ Ollama integration is currently disabled. Using Gemini instead.")
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         st.error("❌ GOOGLE_API_KEY not found. Please add it to your environment.")
#         st.stop()
#     return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# if "llm" not in st.session_state:
#     st.session_state.llm = init_llm(model_choice)

# if st.sidebar.button("🔌 Connect Database"):
#     try:
#         conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
#         st.session_state.db = SQLDatabase.from_uri(conn_str)
#         st.success("✅ Database Connected!")
#     except Exception as e:
#         st.error(f"Failed to connect: {e}")

# # -------------------------------------
# # SQL Agent Setup
# # -------------------------------------
# def get_sql_agent():
#     if "sql_agent" not in st.session_state:
#         try:
#             st.session_state.sql_agent = create_sql_agent(
#                 llm=st.session_state.llm,
#                 db=st.session_state.db,
#                 verbose=True
#             )
#         except Exception as e:
#             st.error(f"SQL Agent Error: {e}")
#     return st.session_state.sql_agent

# # -------------------------------------
# # RAG Setup
# # -------------------------------------
# def connect_pg():
#     return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)

# def fetch_texts(table, col):
#     conn = connect_pg()
#     cur = conn.cursor(cursor_factory=RealDictCursor)
#     cur.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL")
#     data = [r[col] for r in cur.fetchall()]
#     cur.close()
#     conn.close()
#     return data

# def build_rag_index(texts):
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(texts, show_progress_bar=False)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(np.array(embeddings, dtype=np.float32))
#     return index, model

# def query_rag(query):
#     if "rag_index" not in st.session_state:
#         st.warning("⚠️ No RAG index found. Please build one first.")
#         return ""
#     q_vec = st.session_state.rag_model.encode([query])
#     D, I = st.session_state.rag_index.search(np.array(q_vec, dtype=np.float32), len(st.session_state.rag_texts))
#     retrieved = [st.session_state.rag_texts[i] for i in I[0]]
#     context = "\n---\n".join(retrieved)
#     llm = init_llm(model_choice)
#     prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {query}"
#     response = llm.invoke(prompt)
#     return response.content if hasattr(response, "content") else response

# # -------------------------------------
# # Hybrid Router
# # -------------------------------------
# def detect_query_type(query: str) -> str:
#     sql_keywords = [
#         "count", "average", "total", "sum", "list", "how many", "show", "table", "column",
#         "group by", "where", "filter", "join"
#     ]
#     if any(kw in query.lower() for kw in sql_keywords):
#         return "SQL"
#     return "RAG"

# # -------------------------------------
# # Interface per Mode
# # -------------------------------------
# if mode == "SQL Agent 🧾":
#     st.subheader("🧾 SQL Agent Mode")
#     query = st.text_area("Ask something about your data:")
#     if st.button("Run Query"):
#         agent = get_sql_agent()
#         with st.spinner("Generating SQL and fetching results..."):
#             result = agent.invoke(query)
#             st.write(result["output"])

# elif mode == "RAG 📚":
#     st.subheader("📚 RAG Mode")
#     table = st.text_input("Table Name", "your_table")
#     column = st.text_input("Text Column", "description")

#     if st.button("📄 Build RAG Index"):
#         with st.spinner("Building RAG Index..."):
#             texts = fetch_texts(table, column)
#             index, model = build_rag_index(texts)
#             st.session_state.rag_index = index
#             st.session_state.rag_model = model
#             st.session_state.rag_texts = texts
#             st.success(f"Indexed {len(texts)} text records!")

#     st.divider()
#     query = st.text_area("Ask a question:")
#     if st.button("Ask (RAG)") and "rag_index" in st.session_state:
#         with st.spinner("Thinking..."):
#             answer = query_rag(query)
#             st.write(answer)

# elif mode == "Hybrid 🧠":
#     st.subheader("🧠 Hybrid Smart Mode")

#     query = st.text_area("Ask anything (the model decides best approach):")
#     if st.button("Ask (Hybrid)"):
#         with st.spinner("Analyzing your query..."):
#             query_type = detect_query_type(query)
#             st.info(f"🤖 Detected Query Type: {query_type}")

#             if query_type == "SQL":
#                 agent = get_sql_agent()
#                 result = agent.invoke(query)
#                 st.success("💾 SQL Mode Result:")
#                 st.write(result["output"])
#             else:
#                 answer = query_rag(query)
#                 st.success("📚 RAG Mode Result:")
#                 st.write(answer)

# st.sidebar.markdown("---")
# st.sidebar.caption("⚡ Hybrid AI Assistant with Gemini. Ollama option reserved for future use.")