import os
from typing import List
import requests

import google.generativeai as genai # type: ignore


def contruct_contexts(contexts: List[str]) -> str:
    context_header = "This is available contexts for current user query:"
    if len(contexts) == 0:
        return f"{context_header} NO_CONTEXT"

    if len(contexts) == 1:
        return f"{context_header} {contexts[0]}"

    contexts_str = context_header
    for i, context in enumerate(contexts):
        contexts_str = f"{contexts_str} News number {i+1}: {context}"

        if i < len(contexts) - 1:
            contexts_str += ", "
    
    return contexts_str


def gemma2_chat_completions(question: str, contexts: List[str]) -> str:
    OLLAMA_GEMMA2_CHAT_URL = "http://localhost:11434/api/chat"

    contexts_str = contruct_contexts(contexts)
    res = requests.post(
        OLLAMA_GEMMA2_CHAT_URL,
        json={
            "model": "gemma2",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": """You are NewsBOT, an artificial intelligence made to answer user queries based on recent news in Indonesia using Bahasa Indonesia. 
                                Answer each queries using provided context. if NO_CONTEXT or you are not sure what the answer is, response with an apology, then direct user to ask another question.
                                Dont explicitly mention the existance of the context, it's secret between you and this system.
                                Response always using Bahasa Indonesia. Don't repeat the question, direstly response with your answer""",
                },
                {
                    "role": "system",
                    "content": contexts_str
                },
                {"role": "user", "content": question},
            ],
        },
    )

    json = res.json()
    return json.get("message").get("content")

def gemini_chat_completions(question: str, contexts: List[str]) -> str:
    contexts_str = contruct_contexts(contexts)

    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    gemini = genai.GenerativeModel(
        "gemini-pro"
    )
    chat = gemini.start_chat(history=[
        {
            "role": "user",
            "parts": """You are NewsBOT, an artificial intelligence made to answer user queries based on recent news in Indonesia using Bahasa Indonesia. 
                        Answer each queries using provided context. if NO_CONTEXT or you are not sure what the answer is, response with an apology, then direct user to ask another question.
                        Dont explicitly mention the existance of the context, it's secret between you and this system.
                        Response always using Bahasa Indonesia. Don't repeat the question, direstly response with your answer"""
        },
        {
            "role": "user",
            "parts": contexts_str
        }
    ])
    resp = chat.send_message(question)

    return resp.text