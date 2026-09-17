"""Tests for index operation utilities."""

import time
from collections.abc import Generator

import pytest
from pymongo import MongoClient
from pymongo.collection import Collection

from pymongo_search_utils.index import (
    create_fulltext_search_index,
    create_vector_search_index,
    drop_search_index,
    is_index_ready,
    vector_search_index_definition,
    wait_for_docs_in_index,
    wait_for_fulltext_docs_in_index,
    wait_for_predicate,
)

DBNAME = "pymongo_search_utils_test"
COLLECTION_NAME = "test_index"
VECTOR_INDEX_NAME = "vector_index"
FULLTEXT_INDEX_NAME = "fulltext_index"
# A list exercises the field: str | list[str] branch of create_fulltext_search_index.
FULLTEXT_FIELDS = ["page_content", "title"]

TIMEOUT = 120  # TODO - This is a bitter pill to swallow
DIMENSIONS = 10


@pytest.fixture(scope="module")
def collection(client: MongoClient) -> Generator:
    if COLLECTION_NAME not in client[DBNAME].list_collection_names():
        clxn = client[DBNAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DBNAME][COLLECTION_NAME]
    clxn.delete_many({})
    yield clxn
    clxn.delete_many({})


def test_vector_search_index_definition() -> None:
    """Test the vector_search_index_definition helper function."""
    # Test basic definition without filters
    definition = vector_search_index_definition(
        dimensions=128, path="embedding", similarity="cosine"
    )
    assert "fields" in definition
    assert len(definition["fields"]) == 1
    assert definition["fields"][0]["numDimensions"] == 128
    assert definition["fields"][0]["path"] == "embedding"
    assert definition["fields"][0]["similarity"] == "cosine"
    assert definition["fields"][0]["type"] == "vector"

    # Test with filters
    definition = vector_search_index_definition(
        dimensions=256, path="vector", similarity="euclidean", filters=["category", "status"]
    )
    assert len(definition["fields"]) == 3
    assert definition["fields"][0]["type"] == "vector"
    assert definition["fields"][1]["type"] == "filter"
    assert definition["fields"][1]["path"] == "category"
    assert definition["fields"][2]["type"] == "filter"
    assert definition["fields"][2]["path"] == "status"

    # Test with vector_index_options
    definition = vector_search_index_definition(
        dimensions=512,
        path="embed",
        similarity="dotProduct",
        vector_index_options={"quantization": {"type": "scalar"}},
    )
    assert definition["fields"][0]["quantization"] == {"type": "scalar"}

    # Test with kwargs
    definition = vector_search_index_definition(
        dimensions=64, path="vec", similarity="cosine", storedSource=True
    )
    assert definition["storedSource"] is True


def test_vector_search_index_definition_for_autoembedding() -> None:
    """Test autoembedding config."""
    definition = vector_search_index_definition(
        dimensions=-1, path="text", similarity=None, auto_embedding_model="voyage-4"
    )
    assert "fields" in definition
    assert len(definition["fields"]) == 1
    assert definition["fields"][0]["type"] == "autoEmbed"
    assert definition["fields"][0]["path"] == "text"
    assert definition["fields"][0]["model"] == "voyage-4"
    assert definition["fields"][0]["modality"] == "text"

    # Test bad config
    with pytest.raises(ValueError):
        vector_search_index_definition(
            dimensions=64, path="text", similarity=None, auto_embedding_model="voyage-4"
        )
    with pytest.raises(ValueError):
        vector_search_index_definition(
            dimensions=-1,
            path="text",
            similarity=None,
        )
    with pytest.raises(ValueError):
        vector_search_index_definition(
            dimensions=64, path="text", similarity="cosine", auto_embedding_model="voyage-4"
        )
    with pytest.raises(ValueError):
        vector_search_index_definition(
            dimensions=-1,
            path="text",
            similarity="cosine",
        )


def test_wait_for_predicate() -> None:
    """Test the wait_for_predicate utility function."""
    # Test successful predicate
    counter = {"value": 0}

    def increment_predicate():
        counter["value"] += 1
        return counter["value"] >= 3

    start = time.monotonic()
    wait_for_predicate(increment_predicate, "Should not timeout", timeout=5, interval=0.1)
    elapsed = time.monotonic() - start
    assert counter["value"] >= 3
    assert elapsed < 5

    # Test timeout
    def always_false():
        return False

    with pytest.raises(TimeoutError, match="Predicate failed"):
        wait_for_predicate(always_false, "Predicate failed", timeout=0.5, interval=0.1)


def test_wait_for_docs_in_index_nonexistent(
    collection: Collection,
) -> None:
    """Test wait_for_docs_in_index raises error for non-existent index."""

    # Should raise ValueError for non-existent index
    with pytest.raises(ValueError, match="does not exist"):
        wait_for_docs_in_index(collection, "nonexistent_index", 1)


def test_wait_for_fulltext_docs_in_index_raises_on_timeout(collection: Collection) -> None:
    """A timeout raises rather than returning a value a caller could ignore."""
    with pytest.raises(TimeoutError, match="did not index 99 documents"):
        wait_for_fulltext_docs_in_index(
            collection, FULLTEXT_INDEX_NAME, "text", n_docs=99, timeout=3
        )


def test_indexes(collection: Collection, requires_search) -> None:
    """Tests, create, wait, and drop index functions together."""

    # Clean up existing indexes
    for index_info in collection.list_search_indexes():
        drop_search_index(collection, index_info["name"], wait_until_complete=TIMEOUT)
    assert len(collection.list_search_indexes().to_list()) == 0

    # Create vector search index
    create_vector_search_index(
        collection=collection,
        index_name=VECTOR_INDEX_NAME,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity="cosine",
        wait_until_complete=TIMEOUT,
    )

    # Verify index was created
    assert is_index_ready(collection, VECTOR_INDEX_NAME)
    indexes = list(collection.list_search_indexes())
    assert len(indexes) == 1

    # Insert test documents with mocked embeddings
    n_docs = 5
    docs = [{"embedding": [0.1] * DIMENSIONS, "page_content": f"doc {i}"} for i in range(n_docs)]
    collection.insert_many(docs)

    # Wait for documents to be indexed
    assert wait_for_docs_in_index(collection, VECTOR_INDEX_NAME, n_docs)

    # Create fulltext search index
    create_fulltext_search_index(
        collection=collection,
        index_name=FULLTEXT_INDEX_NAME,
        field=FULLTEXT_FIELDS,
        wait_until_complete=TIMEOUT,
    )

    # Verify index was created
    assert is_index_ready(collection, FULLTEXT_INDEX_NAME)
    indexes = collection.list_search_indexes().to_list()
    assert len(indexes) == 2

    # Wait for documents to be indexed
    assert wait_for_fulltext_docs_in_index(collection, FULLTEXT_INDEX_NAME, "page_content")

    # `field` also accepts a single string, the form most callers use. A stored
    # definition is readable as soon as the index is created, so checking this
    # branch needs no readiness wait, which is the expensive part on Atlas.
    create_fulltext_search_index(
        collection=collection, index_name="fulltext_single_field", field="description"
    )
    single = collection.list_search_indexes("fulltext_single_field").try_next()
    assert single is not None
    assert single["latestDefinition"]["mappings"]["dynamic"] is False
    assert set(single["latestDefinition"]["mappings"]["fields"]) == {"description"}
    drop_search_index(collection, "fulltext_single_field", wait_until_complete=TIMEOUT)

    # Verify index values
    for idx in indexes:
        if idx["name"] == VECTOR_INDEX_NAME:
            assert idx["latestDefinition"]["fields"][0]["type"] == "vector"
            assert idx["latestDefinition"]["fields"][0]["path"] == "embedding"
            assert idx["latestDefinition"]["fields"][0]["similarity"] == "cosine"
            assert idx["latestDefinition"]["fields"][0]["numDimensions"] == DIMENSIONS
        elif idx["name"] == FULLTEXT_INDEX_NAME:
            # Community mongot omits `type` from $listSearchIndexes; "search" is
            # the default when it is absent.
            assert idx.get("type", "search") == "search"
            assert idx["latestDefinition"]["mappings"]["dynamic"] is False
            assert set(idx["latestDefinition"]["mappings"]["fields"]) == set(FULLTEXT_FIELDS)
        else:
            raise AssertionError(f"Unexpected index name: {idx['name']}")

    # TODO: Test that we can update the index
    #   "collection.update_vector_search_index requires [https://jira.mongodb.org/browse/DRIVERS-3078]"
    """
    similarity_new = "euclidean"
    update_vector_search_index(
        collection=collection,
        index_name=VECTOR_INDEX_NAME,
        dimensions=DIMENSIONS,
        path="embedding",
        similarity=similarity_new,
        wait_until_complete=TIMEOUT,
    )
    assert is_index_ready(collection, VECTOR_INDEX_NAME)
    assert len(collection.list_search_indexes().to_list()) == 2
    """
    # Drop an index and verify one remains
    drop_search_index(collection, VECTOR_INDEX_NAME, wait_until_complete=5)
    assert [i["name"] for i in collection.list_search_indexes()] == [FULLTEXT_INDEX_NAME]


def test_wait_for_fulltext_docs_in_index_on_empty_collection(client) -> None:
    """Test case when collection has 0 documents."""
    db = client[DBNAME]
    if "empty" not in db.list_collection_names():
        empty_clxn = db.create_collection("empty")
    else:
        empty_clxn = db["empty"]

    # Create fulltext search index
    create_fulltext_search_index(
        collection=empty_clxn,
        index_name=FULLTEXT_INDEX_NAME,
        field=FULLTEXT_FIELDS,
        wait_until_complete=TIMEOUT,
    )
    # Wait for documents to be indexed
    assert wait_for_fulltext_docs_in_index(empty_clxn, FULLTEXT_INDEX_NAME, "page_content")
