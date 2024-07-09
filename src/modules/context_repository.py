from typing import List

import pypika as pk  # type: ignore
from psycopg2 import ProgrammingError

class NewsContext:
    def __init__(
        self, url: str, title: str, content: str, date: str, distance: float
    ) -> None:
        self.url = url
        self.title = title
        self.content = content
        self.date = date
        self.distance = distance

    def __repr__(self):
        return f"{self.title} ({self.url}) [{self.distance:.2f}]"

def get_news_context(
        db, question_embedding: str, max_length=2, max_distance=0.35
    ) -> List[NewsContext]:
        news_table = pk.Table("news")
        cosine_distance = pk.CustomFunction("cosine_distance", ["vector", "vector"])
        distance = cosine_distance(news_table.embedding, pk.Parameter("%s"))

        query = (
            pk.PostgreSQLQuery.from_(news_table)
            .select(
                "url",
                "title",
                "content",
                "date",
                distance.as_("distance"),
            )
            .where(distance <= max_distance)
            .orderby(distance, order=pk.Order.asc)
            .limit(max_length)
        )

        cursor = db.cursor()
        cursor.execute(
            str(query), (question_embedding, question_embedding, question_embedding)
        )

        try:
            results = cursor.fetchall()
        except ProgrammingError:
            results = []
        finally:
            cursor.close()

        if len(results) == 0:
            return []

        return [NewsContext(*result) for result in results]

def filter_content_only(news_contexts: List[NewsContext]) -> List[str]:
     return [news.content for news in news_contexts]