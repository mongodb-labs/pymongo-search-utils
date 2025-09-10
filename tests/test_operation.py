"""Tests for operation utilities."""

import os
from unittest.mock import Mock

import pytest
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

from pymongo_vectorsearch_utils.operation import bulk_embed_and_insert_texts

DB_NAME = "vectorsearch_utils_test"
COLLECTION_NAME = "test_operation"


@pytest.fixture(scope="module")
def client():
    conn_str = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017?directConnection=true")
    client = MongoClient(conn_str)
    yield client
    client.close()


@pytest.fixture
def collection(client):
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]
    clxn.delete_many({})
    yield clxn
    clxn.delete_many({})


@pytest.fixture
def mock_embedding_func():
    """Mock embedding function that returns predictable embeddings."""

    def embedding_func(texts):
        return [[float(i), float(i) * 0.5, float(i) * 0.25] for i in range(len(texts))]

    return embedding_func


class TestBulkEmbedAndInsertTexts:
    def test_empty_texts_returns_empty_list(self, collection: Collection, mock_embedding_func):
        result = bulk_embed_and_insert_texts(
            texts=[],
            metadatas=[],
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
        )
        assert result == []

    def test_basic_insertion_with_generated_ids(self, collection: Collection, mock_embedding_func):
        texts = ["text one", "text two"]
        metadatas = [{"category": "test_1"}, {"category": "test_2"}]

        result = bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="content",
            embedding_key="vector",
        )

        assert len(result) == 2
        assert all(isinstance(id_str, str) for id_str in result)

        docs = list(collection.find({}))
        assert len(docs) == 2

        for i, doc in enumerate(docs):
            assert doc["content"] == texts[i]
            assert doc["vector"] == [float(i), float(i) * 0.5, float(i) * 0.25]
            assert doc["category"] == metadatas[i]["category"]
            assert isinstance(doc["_id"], ObjectId)

    def test_insertion_with_custom_ids(self, collection: Collection, mock_embedding_func):
        texts = ["text one"]
        metadatas = [{"type": "custom"}]
        custom_ids = ["custom_id_123"]

        result = bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
            ids=custom_ids,
        )

        assert result == custom_ids

        doc = collection.find_one({"_id": "custom_id_123"})
        assert doc is not None
        assert doc["text"] == texts[0]
        assert doc["type"] == "custom"

    def test_insertion_with_objectid_string_ids(self, collection: Collection, mock_embedding_func):
        texts = ["text one"]
        metadatas = [{"test": True}]
        object_id_str = str(ObjectId())

        result = bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
            ids=[object_id_str],
        )

        assert result == [object_id_str]

        # Verify document was inserted with ObjectId
        doc = collection.find_one({})
        assert doc is not None
        assert isinstance(doc["_id"], ObjectId)
        assert str(doc["_id"]) == object_id_str

    def test_upsert_behavior(self, collection: Collection, mock_embedding_func):
        texts = ["text one"]
        metadatas = [{"version": 1}]
        custom_id = "upsert_id"

        # First insertion
        bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
            ids=[custom_id],
        )

        new_metadatas = [{"version": 2}]
        bulk_embed_and_insert_texts(
            texts=["updated text"],
            metadatas=new_metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
            ids=[custom_id],
        )

        docs = list(collection.find({}))
        assert len(docs) == 1
        assert docs[0]["text"] == "updated text"
        assert docs[0]["version"] == 2

    def test_with_generator_metadata(self, collection: Collection, mock_embedding_func):
        def metadata_generator():
            yield {"index": 0}
            yield {"index": 1}

        result = bulk_embed_and_insert_texts(
            texts=["text one", "text two"],
            metadatas=metadata_generator(),
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
        )

        assert len(result) == 2
        docs = list(collection.find({}).sort("index", 1))
        assert len(docs) == 2
        assert docs[0]["text"] == "text one"
        assert docs[1]["text"] == "text two"

    def test_embedding_function_called_correctly(self, collection: Collection):
        texts = ["text one", "text two", "text three"]
        metadatas = [{}, {}, {}]

        mock_embedding_func = Mock(return_value=[[1.0], [2.0], [3.0]])

        bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
        )

        mock_embedding_func.assert_called_once_with(texts)

    def test_large_batch_processing(self, collection: Collection, mock_embedding_func):
        num_docs = 100
        texts = [f"text {i}" for i in range(num_docs)]
        metadatas = [{"doc_num": i} for i in range(num_docs)]

        result = bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
        )

        assert len(result) == num_docs
        assert collection.count_documents({}) == num_docs

    def test_with_additional_kwargs(self, collection: Collection, mock_embedding_func):
        texts = ["text one"]
        metadatas = [{}]

        result = bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
            extra_param="ignored",
        )

        assert len(result) == 1

    def test_mismatched_lengths_handled_gracefully(
        self, collection: Collection, mock_embedding_func
    ):
        texts = ["text one", "text two"]
        metadatas = [{"meta": 1}]  # Shorter than texts

        result = bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="text",
            embedding_key="embedding",
        )

        assert len(result) == 1
        docs = list(collection.find({}))
        assert len(docs) == 1
        assert docs[0]["text"] == "text one"

    def test_custom_field_names(self, collection: Collection, mock_embedding_func):
        texts = ["text one"]
        metadatas = [{}]

        bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="content",
            embedding_key="vector",
        )

        doc = collection.find_one({})
        assert doc is not None
        assert "content" in doc
        assert "vector" in doc
        assert doc["content"] == texts[0]
        assert doc["vector"] == [0.0, 0.0, 0.0]
