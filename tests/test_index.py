"""Tests for index operation utilities."""

import os
from collections.abc import Generator

import pytest
from pymongo import MongoClient
from pymongo.collection import Collection

from pymongo_search_utils.index import (
    create_vector_search_index,
    drop_vector_search_index,
    is_index_ready,
    update_vector_search_index,
)

DB_NAME = "pymongo_search_utils_test"
COLLECTION_NAME = "test_index"
VECTOR_INDEX_NAME = "vector_index"

TIMEOUT = 120
DIMENSIONS = 10


@pytest.fixture(scope="module")
def client() -> Generator[MongoClient, None, None]:
    conn_str = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017?directConnection=true")
    client = MongoClient(conn_str)
    yield client
    client.close()


@pytest.fixture
def collection(client) -> Generator:
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]
    clxn.delete_many({})
    yield clxn
    clxn.delete_many({})


def test_search_index_create_and_drop(collection: Collection) -> None:
    index_name = VECTOR_INDEX_NAME
    dimensions = DIMENSIONS
    path = "embedding"
    similarity = "cosine"
    filters: list[str] | None = None
    wait_until_complete = TIMEOUT

    for index_info in collection.list_search_indexes():
        drop_vector_search_index(
            collection, index_info["name"], wait_until_complete=wait_until_complete
        )

    assert len(list(collection.list_search_indexes())) == 0

    create_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=dimensions,
        path=path,
        similarity=similarity,
        filters=filters,
        wait_until_complete=wait_until_complete,
    )

    assert is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name

    drop_vector_search_index(collection, index_name, wait_until_complete=wait_until_complete)

    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 0


@pytest.mark.skip(
    "collection.update_vector_search_index requires [https://jira.mongodb.org/browse/DRIVERS-3078]"
)
def test_search_index_update_vector_search_index(collection: Collection) -> None:
    index_name = "INDEX_TO_UPDATE"
    similarity_orig = "cosine"
    similarity_new = "euclidean"

    create_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_orig,
        wait_until_complete=TIMEOUT,
    )

    assert is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == similarity_orig

    update_vector_search_index(
        collection=collection,
        index_name=index_name,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_new,
        wait_until_complete=TIMEOUT,
    )

    assert is_index_ready(collection, index_name)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1
    assert indexes[0]["name"] == index_name
    assert indexes[0]["latestDefinition"]["fields"][0]["similarity"] == similarity_new
