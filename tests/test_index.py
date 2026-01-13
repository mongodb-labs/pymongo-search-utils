"""Tests for index operation utilities."""

import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from pymongo import MongoClient
from pymongo.collection import Collection

from pymongo_search_utils.index import (
    create_fulltext_search_index,
    create_vector_search_index,
    drop_vector_search_index,
    is_index_ready,
    update_vector_search_index,
    vector_search_index_definition,
    wait_for_docs_in_index,
    wait_for_predicate,
)

DB_NAME = "pymongo_search_utils_test"
COLLECTION_NAME = "test_index"
VECTOR_INDEX_NAME = "vector_index"
FULLTEXT_INDEX_NAME = "fulltext_index"

TIMEOUT = 120
DIMENSIONS = 10

COMMUNITY_WITH_SEARCH = os.environ.get("COMMUNITY_WITH_SEARCH", "")
require_community = pytest.mark.skipif(
    COMMUNITY_WITH_SEARCH == "", reason="Only run in COMMUNITY_WITH_SEARCH is set"
)

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


@pytest.fixture
def mock_search_indexes():
    """Fixture to track mock search indexes."""
    return {}


def test_search_index_create_and_drop(collection: Collection, mock_search_indexes: dict) -> None:
    index_name = VECTOR_INDEX_NAME
    dimensions = DIMENSIONS
    path = "embedding"
    similarity = "cosine"
    filters: list[str] | None = None
    wait_until_complete = TIMEOUT

    def mock_list_search_indexes(name=None):
        if name:
            return [mock_search_indexes[name]] if name in mock_search_indexes else []
        return list(mock_search_indexes.values())

    def mock_create_search_index(model):
        doc = model.document
        mock_search_indexes[doc["name"]] = {
            "name": doc["name"],
            "status": "READY",
            "type": doc.get("type", "search"),
            "latestDefinition": doc["definition"],
        }
        return doc["name"]

    def mock_drop_search_index(name):
        if name in mock_search_indexes:
            del mock_search_indexes[name]

    with (
        patch.object(collection, "list_search_indexes", side_effect=mock_list_search_indexes),
        patch.object(collection, "create_search_index", side_effect=mock_create_search_index),
        patch.object(collection, "drop_search_index", side_effect=mock_drop_search_index),
    ):
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

@require_community
def test_vector_search_index_definition_for_autoembedding() -> None:
    # Test autoembedding config
    definition = vector_search_index_definition(
        dimensions=-1, path="text", similarity=None, auto_embedding_model="voyage-4"
    )
    assert "fields" in definition
    assert len(definition["fields"]) == 1
    assert definition["fields"][0]["type"] == 'autoEmbed'
    assert definition["fields"][0]["path"] == "text"
    assert definition["fields"][0]["model"] == "voyage-4"
    assert definition["fields"][0]["modality"] == "text"

    # Test bad config
    with pytest.raises(ValueError):
        vector_search_index_definition(
            dimensions=64, path="text", similarity=None,
            auto_embedding_model="voyage-4"
        )
    with pytest.raises(ValueError):
        vector_search_index_definition(
            dimensions=-1, path="text", similarity=None,
        )
    with pytest.raises(ValueError):
        vector_search_index_definition(
            dimensions=64, path="text", similarity="cosine",
            auto_embedding_model="voyage-4"
        )
    with pytest.raises(ValueError):
        vector_search_index_definition(
            dimensions=-1, path="text", similarity="cosine",
        )


def test_wait_for_predicate() -> None:
    """Test the wait_for_predicate utility function."""
    import time

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


def test_create_fulltext_search_index_single_field(
    collection: Collection, mock_search_indexes: dict
) -> None:
    """Test creating a fulltext search index on a single field."""
    index_name = FULLTEXT_INDEX_NAME
    field = "description"
    wait_until_complete = TIMEOUT

    def mock_list_search_indexes(name=None):
        if name:
            return [mock_search_indexes[name]] if name in mock_search_indexes else []
        return list(mock_search_indexes.values())

    def mock_create_search_index(model):
        doc = model.document
        mock_search_indexes[doc["name"]] = {
            "name": doc["name"],
            "status": "READY",
            "type": doc.get("type", "search"),
            "latestDefinition": doc["definition"],
        }
        return doc["name"]

    def mock_drop_search_index(name):
        if name in mock_search_indexes:
            del mock_search_indexes[name]

    with (
        patch.object(collection, "list_search_indexes", side_effect=mock_list_search_indexes),
        patch.object(collection, "create_search_index", side_effect=mock_create_search_index),
        patch.object(collection, "drop_search_index", side_effect=mock_drop_search_index),
    ):
        # Clean up existing indexes
        for index_info in collection.list_search_indexes():
            drop_vector_search_index(
                collection, index_info["name"], wait_until_complete=wait_until_complete
            )

        # Create fulltext search index
        create_fulltext_search_index(
            collection=collection,
            index_name=index_name,
            field=field,
            wait_until_complete=wait_until_complete,
        )

        # Verify index was created
        assert is_index_ready(collection, index_name)
        indexes = list(collection.list_search_indexes())
        assert len(indexes) == 1
        assert indexes[0]["name"] == index_name
        assert indexes[0]["type"] == "search"
        assert indexes[0]["latestDefinition"]["mappings"]["dynamic"] is False
        assert field in indexes[0]["latestDefinition"]["mappings"]["fields"]

        # Clean up
        drop_vector_search_index(collection, index_name, wait_until_complete=wait_until_complete)


def test_create_fulltext_search_index_multiple_fields(
    collection: Collection, mock_search_indexes: dict
) -> None:
    """Test creating a fulltext search index on multiple fields."""
    index_name = "fulltext_multi_index"
    fields = ["title", "description", "content"]
    wait_until_complete = TIMEOUT

    def mock_list_search_indexes(name=None):
        if name:
            return [mock_search_indexes[name]] if name in mock_search_indexes else []
        return list(mock_search_indexes.values())

    def mock_create_search_index(model):
        doc = model.document
        mock_search_indexes[doc["name"]] = {
            "name": doc["name"],
            "status": "READY",
            "type": doc.get("type", "search"),
            "latestDefinition": doc["definition"],
        }
        return doc["name"]

    def mock_drop_search_index(name):
        if name in mock_search_indexes:
            del mock_search_indexes[name]

    with (
        patch.object(collection, "list_search_indexes", side_effect=mock_list_search_indexes),
        patch.object(collection, "create_search_index", side_effect=mock_create_search_index),
        patch.object(collection, "drop_search_index", side_effect=mock_drop_search_index),
    ):
        # Clean up existing indexes
        for index_info in collection.list_search_indexes():
            drop_vector_search_index(
                collection, index_info["name"], wait_until_complete=wait_until_complete
            )

        # Create fulltext search index with multiple fields
        create_fulltext_search_index(
            collection=collection,
            index_name=index_name,
            field=fields,
            wait_until_complete=wait_until_complete,
        )

        # Verify index was created
        assert is_index_ready(collection, index_name)
        indexes = list(collection.list_search_indexes())
        assert len(indexes) == 1
        assert indexes[0]["name"] == index_name
        assert indexes[0]["type"] == "search"

        # Verify all fields are in the index
        index_fields = indexes[0]["latestDefinition"]["mappings"]["fields"]
        for field in fields:
            assert field in index_fields
            assert index_fields[field] == [{"type": "string"}]

        # Clean up
        drop_vector_search_index(collection, index_name, wait_until_complete=wait_until_complete)


def test_wait_for_docs_in_index(collection: Collection, mock_search_indexes: dict) -> None:
    """Test waiting for documents to be indexed in a vector search index."""
    index_name = "wait_docs_index"
    dimensions = DIMENSIONS
    path = "embedding"
    similarity = "cosine"
    wait_until_complete = TIMEOUT

    def mock_list_search_indexes(name=None):
        if name:
            result = [mock_search_indexes[name]] if name in mock_search_indexes else []

            # When called with a name parameter (e.g., for is_index_ready), return an iterable
            # When called with a name parameter for wait_for_docs_in_index, return a mock cursor
            # We need to check the context - if it has to_list, it's for wait_for_docs_in_index
            class MockIndexCursor:
                def __init__(self, data):
                    self.data = data

                def __iter__(self):
                    return iter(self.data)

                def to_list(self):
                    return self.data

            return MockIndexCursor(result)
        return list(mock_search_indexes.values())

    def mock_create_search_index(model):
        doc = model.document
        mock_search_indexes[doc["name"]] = {
            "name": doc["name"],
            "status": "READY",
            "type": doc.get("type", "search"),
            "latestDefinition": doc["definition"],
        }
        return doc["name"]

    def mock_drop_search_index(name):
        if name in mock_search_indexes:
            del mock_search_indexes[name]

    # Insert test documents with embeddings
    n_docs = 5
    docs = [{"_id": i, path: [0.1] * dimensions, "text": f"doc {i}"} for i in range(n_docs)]
    collection.insert_many(docs)

    def mock_aggregate(pipeline):
        # Return mock results for vector search aggregation
        mock_cursor = MagicMock()
        mock_cursor.to_list.return_value = [{"_id": i, "search_score": 0.9} for i in range(n_docs)]
        return mock_cursor

    with (
        patch.object(collection, "list_search_indexes", side_effect=mock_list_search_indexes),
        patch.object(collection, "create_search_index", side_effect=mock_create_search_index),
        patch.object(collection, "drop_search_index", side_effect=mock_drop_search_index),
        patch.object(collection, "aggregate", side_effect=mock_aggregate),
    ):
        # Clean up existing indexes
        for index_info in list(mock_search_indexes.values()):
            drop_vector_search_index(
                collection, index_info["name"], wait_until_complete=wait_until_complete
            )

        # Create vector search index
        create_vector_search_index(
            collection=collection,
            index_name=index_name,
            dimensions=dimensions,
            path=path,
            similarity=similarity,
            wait_until_complete=wait_until_complete,
        )

        # Wait for documents to be indexed
        result = wait_for_docs_in_index(collection, index_name, n_docs)
        assert result is True

        # Clean up
        drop_vector_search_index(collection, index_name, wait_until_complete=wait_until_complete)


def test_wait_for_docs_in_index_nonexistent(
    collection: Collection, mock_search_indexes: dict
) -> None:
    """Test wait_for_docs_in_index raises error for non-existent index."""

    def mock_list_search_indexes(name=None):
        if name:
            result = [mock_search_indexes[name]] if name in mock_search_indexes else []

            # Return a mock cursor with to_list method
            class MockIndexCursor:
                def __init__(self, data):
                    self.data = data

                def __iter__(self):
                    return iter(self.data)

                def to_list(self):
                    return self.data

            return MockIndexCursor(result)
        return list(mock_search_indexes.values())

    def mock_drop_search_index(name):
        if name in mock_search_indexes:
            del mock_search_indexes[name]

    with (
        patch.object(collection, "list_search_indexes", side_effect=mock_list_search_indexes),
        patch.object(collection, "drop_search_index", side_effect=mock_drop_search_index),
    ):
        # Ensure no indexes exist
        for index_info in list(mock_search_indexes.values()):
            drop_vector_search_index(collection, index_info["name"], wait_until_complete=TIMEOUT)

        # Should raise ValueError for non-existent index
        with pytest.raises(ValueError, match="does not exist"):
            wait_for_docs_in_index(collection, "nonexistent_index", 1)
