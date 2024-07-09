import polars as pl
import httpx
import asyncio

df = pl.read_csv("~/Documents/projects/indonesian_news_datasets/data.csv")
print("Done reading ...")

async def get_word_embedding(text: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore, httpx.AsyncClient() as client:
        OLLAMA_NOMIC_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
        res = await client.post(
            OLLAMA_NOMIC_EMBEDDING_URL, 
            json={
                    "model": "nomic-embed-text",
                    "prompt": text
                }
            )

        json = res.json()

        return str(json.get("embedding"))

async def populate_async_column():
    semaphore = asyncio.Semaphore(10)
    nomic_embedding = await asyncio.gather(*[get_word_embedding(row['content'], semaphore) for row in df.iter_rows(named=True)])
    return nomic_embedding
    

print("Running asyncio ...")
nomic_embedding = asyncio.run(populate_async_column())
print("Done Running asyncio ...")

df = df.with_columns(nomic_embedding=pl.Series(nomic_embedding, dtype=pl.String))
df.write_csv("~/Documents/projects/indonesian_news_datasets/data_with_nomic_embeddings.csv")