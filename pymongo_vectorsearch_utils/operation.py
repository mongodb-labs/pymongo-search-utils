"""CRUD utilities and helpers."""

from collections.abc import Callable, Generator, Iterable
from typing import Any

from bson import ObjectId
from pymongo import ReplaceOne
from pymongo.synchronous.collection import Collection

from pymongo_vectorsearch_utils.util import oid_to_str, str_to_oid


def bulk_embed_and_insert_texts(
    texts: list[str] | Iterable[str],
    metadatas: list[dict] | Generator[dict, Any, Any],
    embedding_func: Callable[[list[str]], list[list[float]]],
    collection: Collection[Any],
    text_key: str,
    embedding_key: str,
    ids: list[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    """Bulk insert single batch of texts, embeddings, and optionally ids.

    Important notes on ids:
    - If _id or id is a key in the metadatas dicts, one must
        pop them and provide as separate list.
    - They must be unique.
    - If they are not provided, unique ones are created,
        stored as bson.ObjectIds internally, and strings in the database.
        These will appear in Document.metadata with key, '_id'.

    Args:
        texts: Iterable of strings to add to the vectorstore.
        metadatas: Optional list of metadatas associated with the texts.
        embedding_func: A function that generates embedding vectors from the texts.
        collection: The MongoDB collection where documents will be inserted.
        text_key: The field name where thet text will be stored in each document.
        embedding_key: The field name where the embedding will be stored in each document.
        ids: Optional list of unique ids that will be used as index in VectorStore.
            See note on ids.
    """
    if not texts:
        return []
    # Compute embedding vectors
    embeddings = embedding_func(list(texts))
    if not ids:
        ids = [str(ObjectId()) for _ in range(len(list(texts)))]
    docs = [
        {
            "_id": str_to_oid(i),
            text_key: t,
            embedding_key: embedding,
            **m,
        }
        for i, t, m, embedding in zip(ids, texts, metadatas, embeddings, strict=False)
    ]
    operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in docs]
    # insert the documents in MongoDB Atlas
    result = collection.bulk_write(operations)
    assert result.upserted_ids is not None
    return [oid_to_str(_id) for _id in result.upserted_ids.values()]
