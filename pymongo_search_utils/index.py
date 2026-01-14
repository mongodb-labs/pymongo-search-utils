import logging
from collections.abc import Callable
from time import monotonic, sleep
from typing import Any

from pymongo.operations import SearchIndexModel
from pymongo.synchronous.collection import Collection

TIMEOUT = 120
INTERVAL = 0.5

logger = logging.getLogger(__file__)


def _check_param_config(
    *,
    dimensions: int,
    similarity: str | None,
    auto_embedding_model: str | None,
):
    if auto_embedding_model is not None and (dimensions != -1 or similarity is not None):
        raise ValueError(
            "if auto_embedding_model is set, then neither dimensions nor similarity may be set."
        )
    if auto_embedding_model is None and (dimensions == -1 or similarity is None):
        raise ValueError("please specify dimensions and similarity.")


def vector_search_index_definition(
    path: str,
    dimensions: int,
    similarity: str | None,
    filters: list[str] | None = None,
    vector_index_options: dict | None = None,
    auto_embedding_model: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a vector search index definition.

    Args:
        dimensions (int): The number of dimensions for vector embeddings,
            `None` if using auto-embeddings.
        path (str): The name of the indexed field containing the vector embeddings.
        similarity (Optional[str]): The type of similarity metric to use.
            One of "euclidean", "cosine", or "dotProduct". `None` if using auto-embeddings.
        filters (Optional[List[str]]): If provided, a list of fields to filter on
            in addition to the vector search.
        auto_embedding_model (Optional[str]): The name of the auto embedding model to use,
            `None` if not using auto-embeddings.
        kwargs (Any): Keyword arguments supplying any additional options to the vector search index.

    Returns:
        A dictionary representing the vector search index definition.
    """
    # https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/
    _check_param_config(
        dimensions=dimensions, similarity=similarity, auto_embedding_model=auto_embedding_model
    )

    if auto_embedding_model is not None:
        fields = [
            {
                "type": "autoEmbed",
                "path": path,
                "model": auto_embedding_model,
                "modality": "text",
                **(vector_index_options or {}),
            },
        ]
    else:
        fields = [
            {
                "numDimensions": dimensions,
                "path": path,
                "similarity": similarity,
                "type": "vector",
                **(vector_index_options or {}),
            },
        ]
    if filters:
        for field in filters:
            fields.append({"type": "filter", "path": field})
    definition = {"fields": fields}
    definition.update(kwargs)
    return definition


def is_index_ready(collection: Collection[Any], index_name: str) -> bool:
    """Check for the index name in the list of available search indexes to see if the
    specified index is of status READY

    Args:
        collection (Collection): MongoDB Collection to for the search indexes
        index_name (str): Vector Search Index name

    Returns:
        bool : True if the index is present and READY false otherwise
    """
    for index in collection.list_search_indexes(index_name):
        if index["status"] == "READY":
            return True
    return False


def wait_for_predicate(
    predicate: Callable[..., Any], err: str, timeout: float = TIMEOUT, interval: float = INTERVAL
) -> None:
    """Generic to block until the predicate returns true

    Args:
        predicate (Callable[, bool]): A function that returns a boolean value
        err (str): Error message to raise if nothing occurs
        timeout (float, optional): Wait time for predicate. Defaults to TIMEOUT.
        interval (float, optional): Interval to check predicate. Defaults to DELAY.

    Raises:
        TimeoutError: _description_
    """
    start = monotonic()
    while not predicate():
        if monotonic() - start > timeout:
            raise TimeoutError(err)
        sleep(interval)


def create_vector_search_index(
    collection: Collection[Any],
    index_name: str,
    path: str,
    dimensions: int,
    similarity: str | None,
    filters: list[str] | None = None,
    vector_index_options: dict | None = None,
    *,
    wait_until_complete: float | None = None,
    auto_embedding_model: str | None = None,
    **kwargs: Any,
) -> None:
    """Create a vector search index on the specified field.

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding,
            `None` if using auto-embeddings
        path (str): field with vector embedding
        similarity (Optional[str]): The similarity score used for the index,
            `None` if using auto-embeddings.
        filters (List[str]): Fields/paths to index to allow filtering in $vectorSearch
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
        auto_embedding_model (Optional[str]): The name of the auto embedding model to use,
            `None` if not using auto-embeddings.
        kwargs: Keyword arguments supplying any additional options to SearchIndexModel.
    """
    logger.info("Creating Search Index %s on %s", index_name, collection.name)

    _check_param_config(
        dimensions=dimensions, similarity=similarity, auto_embedding_model=auto_embedding_model
    )

    if collection.name not in collection.database.list_collection_names():
        collection.database.create_collection(collection.name)
    result = collection.create_search_index(
        SearchIndexModel(
            definition=vector_search_index_definition(
                dimensions=dimensions,
                path=path,
                similarity=similarity,
                filters=filters,
                vector_index_options=vector_index_options,
                auto_embedding_model=auto_embedding_model,
                **kwargs,
            ),
            name=index_name,
            type="vectorSearch",
        )
    )

    if wait_until_complete:
        wait_for_predicate(
            predicate=lambda: is_index_ready(collection, index_name),
            err=f"{index_name=} did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info(result)


def update_vector_search_index(
    collection: Collection[Any],
    index_name: str,
    path: str,
    dimensions: int,
    similarity: str | None,
    filters: list[str] | None = None,
    vector_index_options: dict | None = None,
    *,
    wait_until_complete: float | None = None,
    auto_embedding_model: str | None = None,
    **kwargs: Any,
) -> None:
    """Update a search index.

    Replace the existing index definition with the provided definition.

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        dimensions (int): Number of dimensions in embedding,
            `None` if using auto-embeddings.
        path (str): field with vector embedding
        similarity (Optional[str]): The similarity score used for the index,
            `None` if using auto-embeddings.
        filters (List[str]): Fields/paths to index to allow filtering in $vectorSearch
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
        auto_embedding_model (Optional[str]): The name of the auto embedding model to use,
            `None` if not using auto-embeddings.
        kwargs: Keyword arguments supplying any additional options to SearchIndexModel.
    """
    logger.info("Updating Search Index %s from Collection: %s", index_name, collection.name)

    _check_param_config(
        dimensions=dimensions, similarity=similarity, auto_embedding_model=auto_embedding_model
    )

    collection.update_search_index(
        name=index_name,
        definition=vector_search_index_definition(
            dimensions=dimensions,
            path=path,
            similarity=similarity,
            filters=filters,
            vector_index_options=vector_index_options,
            auto_embedding_model=auto_embedding_model,
            **kwargs,
        ),
    )
    if wait_until_complete:
        wait_for_predicate(
            predicate=lambda: is_index_ready(collection, index_name),
            err=f"Index {index_name} update did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info("Update succeeded")


def drop_vector_search_index(
    collection: Collection[Any],
    index_name: str,
    *,
    wait_until_complete: float | None = None,
) -> None:
    """Drop an existing vector search index.

    Args:
        collection (Collection): MongoDB Collection with index to be dropped.
        index_name (str): Name of the MongoDB index.
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready.
    """
    logger.info("Dropping Search Index %s from Collection: %s", index_name, collection.name)
    collection.drop_search_index(index_name)
    if wait_until_complete:
        wait_for_predicate(
            predicate=lambda: len(list(collection.list_search_indexes())) == 0,
            err=f"Index {index_name} did not drop in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info("Vector Search index %s.%s dropped", collection.name, index_name)


def create_fulltext_search_index(
    collection: Collection[Any],
    index_name: str,
    field: str | list[str],
    *,
    wait_until_complete: float | None = None,
    **kwargs: Any,
) -> None:
    """Create a fulltext search index on the specified field(s).

    Args:
        collection (Collection): MongoDB Collection
        index_name (str): Name of Index
        field (str): Field to index
        wait_until_complete (Optional[float]): If provided, number of seconds to wait
            until search index is ready
        kwargs: Keyword arguments supplying any additional options to SearchIndexModel.
    """
    logger.info("Creating Search Index %s on %s", index_name, collection.name)

    if collection.name not in collection.database.list_collection_names():
        collection.database.create_collection(collection.name)

    if isinstance(field, str):
        fields_definition = {field: [{"type": "string"}]}
    else:
        fields_definition = {f: [{"type": "string"}] for f in field}
    definition = {"mappings": {"dynamic": False, "fields": fields_definition}}
    result = collection.create_search_index(
        SearchIndexModel(
            definition=definition,
            name=index_name,
            type="search",
            **kwargs,
        )
    )
    if wait_until_complete:
        wait_for_predicate(
            predicate=lambda: is_index_ready(collection, index_name),
            err=f"{index_name=} did not complete in {wait_until_complete}!",
            timeout=wait_until_complete,
        )
    logger.info(result)


def wait_for_docs_in_index(
    collection: Collection[Any],
    index_name: str,
    n_docs: int,
) -> bool:
    """Wait until the given number of documents are indexed by the given index.

    Args:
        collection (Collection): A MongoDB Collection.
        index_name (str): The name of the index.
        embedding_field (str): The name of the document field containing embeddings.
        n_docs (int): The number of documents to expect in the index.
    """
    indexes = collection.list_search_indexes(index_name).to_list()
    if len(indexes) == 0:
        raise ValueError(f"Index {index_name} does not exist in collection {collection.name}")
    index = indexes[0]
    num_dimensions = index["latestDefinition"]["fields"][0]["numDimensions"]
    field = index["latestDefinition"]["fields"][0]["path"]

    query_vector = [0.001] * num_dimensions  # Dummy vector
    query = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": field,
                "queryVector": query_vector,
                "numCandidates": n_docs,
                "limit": n_docs,
            }
        },
        {"$project": {"_id": 1, "search_score": {"$meta": "vectorSearchScore"}}},
    ]
    start = monotonic()
    while monotonic() - start <= TIMEOUT:
        if len(collection.aggregate(query).to_list()) == n_docs:
            return True
        else:
            sleep(INTERVAL)
    return False
