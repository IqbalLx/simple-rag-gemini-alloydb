import os
from dotenv import load_dotenv
import psycopg2 as pg
import streamlit as st

from modules.word_embedder import get_word_embedding, get_word_embedding_remote
from modules.context_repository import filter_content_only, get_news_context
from modules.llm import gemma2_chat_completions, gemini_chat_completions

if os.path.exists(".env"):
    load_dotenv(".env")

IS_REMOTE_MODE = os.environ["MODE"] == "network"

db = pg.connect(os.environ["POSTGRES_CONNSTRING"])

st.title("💬 NewsBOT")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Halooo"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    question_embeddings = (
        get_word_embedding_remote(question)
        if IS_REMOTE_MODE
        else get_word_embedding(question)
    )
    contexts = get_news_context(db, question_embeddings)
    bot_response = (
        gemini_chat_completions(question, filter_content_only(contexts))
        if IS_REMOTE_MODE
        else gemma2_chat_completions(question, filter_content_only(contexts))
    )

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.chat_message("assistant").write(bot_response)
    st.chat_message("assistant").write(contexts)
