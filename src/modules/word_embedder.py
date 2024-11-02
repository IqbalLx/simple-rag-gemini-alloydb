import os
import requests


def get_word_embedding(word: str) -> str:
    OLLAMA_NOMIC_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
    res = requests.post(
        OLLAMA_NOMIC_EMBEDDING_URL, json={"model": "nomic-embed-text", "prompt": word}
    )

    json = res.json()
    return str(json.get("embedding"))


def get_word_embedding_remote(word: str) -> str:
    NOMICAI_BASE_URL = "https://api-atlas.nomic.ai/v1"
    res = requests.post(
        f"{NOMICAI_BASE_URL}/embedding/text",
        headers={"Authorization": f"Bearer {os.environ['NOMICAI_API_KEY']}"},
        json={"model": "nomic-embed-text-v1.5", "texts": [word], "task_type":'search_query',},
    )

    json = res.json()
    return str(json.get("embeddings")[0])
