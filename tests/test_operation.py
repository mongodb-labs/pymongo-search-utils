"""Tests for operation utilities."""

import os
from unittest.mock import Mock, patch

import pytest
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

from pymongo_search_utils import drop_vector_search_index
from pymongo_search_utils.index import create_vector_search_index, wait_for_docs_in_index
from pymongo_search_utils.operation import bulk_embed_and_insert_texts, execute_search_query

DB_NAME = "pymongo_search_utils_test"
COLLECTION_NAME = "test_operation"
VECTOR_INDEX_NAME = "operation_vector_index"

COMMUNITY_WITH_SEARCH = os.environ.get("COMMUNITY_WITH_SEARCH", "")
require_community = pytest.mark.skipif(
    COMMUNITY_WITH_SEARCH == "", reason="Only run in COMMUNITY_WITH_SEARCH is set"
)

@pytest.fixture(scope="module")
def client():
    conn_str = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017?directConnection=true")
    client = MongoClient(conn_str)
    yield client
    client.close()


@pytest.fixture(scope="module")
def preserved_collection(client):
    if COLLECTION_NAME not in client[DB_NAME].list_collection_names():
        clxn = client[DB_NAME].create_collection(COLLECTION_NAME)
    else:
        clxn = client[DB_NAME][COLLECTION_NAME]
    clxn.delete_many({})
    yield clxn
    clxn.delete_many({})


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

    @require_community
    def test_autoembedding(self, collection: Collection, mock_embedding_func):
        texts = ["text one"]
        metadatas = [{}]

        bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=mock_embedding_func,
            collection=collection,
            text_key="content",
            embedding_key="vector",
            autoembedding=True
        )

        doc = collection.find_one({})
        assert doc is not None
        assert "content" in doc
        assert "vector" not in doc
        assert doc["content"] == texts[0]


class TestExecuteSearchQuery:
    @pytest.fixture(scope="class", autouse=True)
    def vector_search_index(self, client):
        coll = client[DB_NAME][COLLECTION_NAME]
        if len(coll.list_search_indexes(VECTOR_INDEX_NAME).to_list()) == 0:
            create_vector_search_index(
                collection=coll,
                index_name=VECTOR_INDEX_NAME,
                dimensions=3,
                path="embedding",
                similarity="cosine",
                filters=["category", "color", "wheels"],
                wait_until_complete=120,
            )
        yield
        drop_vector_search_index(collection=coll, index_name=VECTOR_INDEX_NAME)

    @pytest.fixture(scope="class", autouse=True)
    def sample_docs(self, preserved_collection: Collection, vector_search_index):
        texts = ["apple fruit", "banana fruit", "car vehicle", "bike vehicle"]
        metadatas = [
            {"category": "fruit", "color": "red"},
            {"category": "fruit", "color": "yellow"},
            {"category": "vehicle", "wheels": 4},
            {"category": "vehicle", "wheels": 2},
        ]

        def embeddings(texts):
            mapping = {
                "apple fruit": [1.0, 0.5, 0.0],
                "banana fruit": [0.5, 0.5, 0.0],
                "car vehicle": [0.0, 0.5, 1.0],
                "bike vehicle": [0.0, 1.0, 0.5],
            }
            return [mapping[text] for text in texts]

        bulk_embed_and_insert_texts(
            texts=texts,
            metadatas=metadatas,
            embedding_func=embeddings,
            collection=preserved_collection,
            text_key="text",
            embedding_key="embedding",
        )
        # Add a document that should not be returned in searches
        preserved_collection.insert_one(
            {
                "category": "fruit",
                "color": "red",
                "embedding": [1.0, 1.0, 1.0],
            }
        )
        wait_for_docs_in_index(preserved_collection, VECTOR_INDEX_NAME, n_docs=5)
        return preserved_collection

    def test_basic_search_query(self, sample_docs: Collection):
        query_vector = [1.0, 0.5, 0.0]

        result = execute_search_query(
            query_vector=query_vector,
            collection=sample_docs,
            embedding_key="embedding",
            text_key="text",
            index_name=VECTOR_INDEX_NAME,
            k=2,
        )

        assert len(result) == 2
        assert result[0]["text"] == "apple fruit"
        assert result[1]["text"] == "banana fruit"
        assert "score" in result[0]
        assert "score" in result[1]

    def test_search_with_pre_filter(self, sample_docs: Collection):
        query_vector = [1.0, 0.5, 1.0]
        pre_filter = {"category": "fruit"}

        result = execute_search_query(
            query_vector=query_vector,
            collection=sample_docs,
            embedding_key="embedding",
            text_key="text",
            index_name=VECTOR_INDEX_NAME,
            k=4,
            pre_filter=pre_filter,
        )

        assert len(result) == 2
        assert result[0]["category"] == "fruit"
        assert result[1]["category"] == "fruit"

    def test_search_with_post_filter_pipeline(self, sample_docs: Collection):
        query_vector = [1.0, 0.5, 0.0]
        post_filter_pipeline = [
            {"$match": {"score": {"$gte": 0.99}}},
            {"$sort": {"score": -1}},
        ]

        result = execute_search_query(
            query_vector=query_vector,
            collection=sample_docs,
            embedding_key="embedding",
            text_key="text",
            index_name=VECTOR_INDEX_NAME,
            k=2,
            post_filter_pipeline=post_filter_pipeline,
        )

        assert len(result) == 1

    def test_search_with_embeddings_included(self, sample_docs: Collection):
        query_vector = [1.0, 0.5, 0.0]

        result = execute_search_query(
            query_vector=query_vector,
            collection=sample_docs,
            embedding_key="embedding",
            text_key="text",
            index_name=VECTOR_INDEX_NAME,
            k=1,
            include_embeddings=True,
        )

        assert len(result) == 1
        assert "embedding" in result[0]
        assert result[0]["embedding"] == [1.0, 0.5, 0.0]

    def test_search_with_custom_field_names(self, sample_docs: Collection):
        query_vector = [1.0, 0.5, 0.25]

        mock_cursor = [
            {
                "_id": ObjectId(),
                "content": "apple fruit",
                "vector": [1.0, 0.5, 0.25],
                "score": 0.9,
            }
        ]

        with patch.object(sample_docs, "aggregate") as mock_aggregate:
            mock_aggregate.return_value = mock_cursor

            result = execute_search_query(
                query_vector=query_vector,
                collection=sample_docs,
                embedding_key="vector",
                text_key="content",
                index_name=VECTOR_INDEX_NAME,
                k=1,
            )

            assert len(result) == 1
            assert "content" in result[0]
            assert result[0]["content"] == "apple fruit"

            pipeline_arg = mock_aggregate.call_args[0][0]
            vector_search_stage = pipeline_arg[0]["$vectorSearch"]
            assert vector_search_stage["path"] == "vector"
            assert {"$project": {"vector": 0}} in pipeline_arg

    def test_search_filters_documents_without_text_key(self, sample_docs: Collection):
        query_vector = [1.0, 0.5, 0.0]

        result = execute_search_query(
            query_vector=query_vector,
            collection=sample_docs,
            embedding_key="embedding",
            text_key="text",
            index_name=VECTOR_INDEX_NAME,
            k=3,
        )

        # Should only return documents with text field
        assert len(result) == 2
        assert all("text" in doc for doc in result)
        assert result[0]["text"] == "apple fruit"
        assert result[1]["text"] == "banana fruit"
