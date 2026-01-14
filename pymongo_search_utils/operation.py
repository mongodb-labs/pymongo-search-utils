"""CRUD utilities and helpers."""

from collections.abc import Callable, Generator, Iterable
from typing import Any

from bson import ObjectId
from pymongo import ReplaceOne
from pymongo.synchronous.collection import Collection

from pymongo_search_utils.pipeline import vector_search_stage
from pymongo_search_utils.util import oid_to_str, str_to_oid


def bulk_embed_and_insert_texts(
    texts: list[str] | Iterable[str],
    metadatas: list[dict] | Generator[dict, Any, Any],
    embedding_func: Callable[[list[str]], list[list[float]]],
    collection: Collection[Any],
    text_key: str,
    embedding_key: str,
    ids: list[str] | None = None,
    autoembedding: bool = False,
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

    if not ids:
        ids = [str(ObjectId()) for _ in range(len(list(texts)))]

    if autoembedding:
        docs = [
            {
                "_id": str_to_oid(i),
                text_key: t,
                **m,
            }
            for i, t, m in zip(ids, texts, metadatas, strict=False)
        ]
    else:
        # Compute embedding vectors
        embeddings = embedding_func(list(texts))
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


def execute_search_query(
    query_vector: list[float],
    collection: Collection[Any],
    embedding_key: str,
    text_key: str,
    index_name: str,
    k: int = 4,
    pre_filter: dict[str, Any] | None = None,
    post_filter_pipeline: list[dict[str, Any]] | None = None,
    oversampling_factor: int = 10,
    include_embeddings: bool = False,
    **kwargs: Any,
) -> list[tuple[Any, float]]:
    """Execute a MongoDB vector search query."""

    # Atlas Vector Search, potentially with filter
    pipeline = [
        vector_search_stage(
            query_vector,
            embedding_key,
            index_name,
            k,
            pre_filter,
            oversampling_factor,
            **kwargs,
        ),
        {"$set": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    # Remove embeddings unless requested.
    if not include_embeddings:
        pipeline.append({"$project": {embedding_key: 0}})
    # Post-processing
    if post_filter_pipeline is not None:
        pipeline.extend(post_filter_pipeline)

    # Execution
    cursor = collection.aggregate(pipeline)
    docs = []

    for doc in cursor:
        if text_key not in doc:
            continue
        docs.append(doc)
    return docs
