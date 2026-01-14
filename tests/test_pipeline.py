"""Tests for pipeline aggregation generator utilities."""

from pymongo_search_utils.pipeline import (
    autoembedding_vector_search_stage,
    combine_pipelines,
    final_hybrid_stage,
    reciprocal_rank_stage,
    text_search_stage,
    vector_search_stage,
)


class TestTextSearchStage:
    def test_basic_text_search(self):
        result = text_search_stage(
            query="test query", search_field="content", index_name="test_index"
        )

        expected = [
            {
                "$search": {
                    "index": "test_index",
                    "text": {"query": "test query", "path": "content"},
                }
            },
            {"$set": {"score": {"$meta": "searchScore"}}},
        ]

        assert result == expected

    def test_text_search_with_multiple_fields(self):
        result = text_search_stage(
            query="test query", search_field=["title", "content"], index_name="test_index"
        )

        assert result[0]["$search"]["text"]["path"] == ["title", "content"]

    def test_text_search_with_filter(self):
        filter_dict = {"category": "tech"}
        result = text_search_stage(
            query="test query", search_field="content", index_name="test_index", filter=filter_dict
        )

        assert {"$match": filter_dict} in result

    def test_text_search_with_limit(self):
        result = text_search_stage(
            query="test query", search_field="content", index_name="test_index", limit=10
        )

        assert {"$limit": 10} in result

    def test_text_search_without_scores(self):
        result = text_search_stage(
            query="test query",
            search_field="content",
            index_name="test_index",
            include_scores=False,
        )

        score_stage = {"$set": {"score": {"$meta": "searchScore"}}}
        assert score_stage not in result

    def test_text_search_with_all_parameters(self):
        filter_dict = {"status": "published"}
        result = text_search_stage(
            query="test query",
            search_field=["title", "description", "content"],
            index_name="test_index",
            limit=20,
            filter=filter_dict,
            include_scores=True,
        )

        assert len(result) == 4
        assert result[0]["$search"]["index"] == "test_index"
        assert result[1] == {"$match": filter_dict}
        assert result[2] == {"$set": {"score": {"$meta": "searchScore"}}}
        assert result[3] == {"$limit": 20}


class TestVectorSearchStage:
    def test_basic_vector_search(self):
        query_vector = [0.1, 0.2, 0.3, 0.4]
        result = vector_search_stage(
            query_vector=query_vector, search_field="embedding", index_name="vector_index"
        )

        expected = {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 40,
                "limit": 4,
            }
        }

        assert result == expected

    def test_vector_search_with_custom_top_k(self):
        query_vector = [0.1, 0.2, 0.3]
        result = vector_search_stage(
            query_vector=query_vector, search_field="embedding", index_name="vector_index", top_k=10
        )

        assert result["$vectorSearch"]["limit"] == 10
        assert result["$vectorSearch"]["numCandidates"] == 100

    def test_vector_search_with_custom_oversampling(self):
        query_vector = [0.1, 0.2, 0.3]
        result = vector_search_stage(
            query_vector=query_vector,
            search_field="embedding",
            index_name="vector_index",
            top_k=5,
            oversampling_factor=20,
        )

        assert result["$vectorSearch"]["numCandidates"] == 100

    def test_vector_search_with_filter(self):
        query_vector = [0.1, 0.2, 0.3]
        filter_dict = {"metadata.category": "science"}
        result = vector_search_stage(
            query_vector=query_vector,
            search_field="embedding",
            index_name="vector_index",
            filter=filter_dict,
        )

        assert result["$vectorSearch"]["filter"] == filter_dict

    def test_vector_search_with_all_parameters(self):
        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        filter_dict = {"published": True, "language": "en"}
        result = vector_search_stage(
            query_vector=query_vector,
            search_field="text_embedding",
            index_name="content_vector_index",
            top_k=15,
            filter=filter_dict,
            oversampling_factor=8,
        )

        expected = {
            "$vectorSearch": {
                "index": "content_vector_index",
                "path": "text_embedding",
                "queryVector": query_vector,
                "numCandidates": 120,
                "limit": 15,
                "filter": filter_dict,
            }
        }

        assert result == expected

    def test_basic_vector_search_autoembedding(self):
        query = "Songs"
        result = autoembedding_vector_search_stage(
            query=query, search_field="text", index_name="vector_index", model="voyage-4"
        )

        expected = {
            "$vectorSearch": {
                "index": "vector_index",
                "model": "voyage-4",
                "path": "text",
                "query": {"text": "Songs"},
                "numCandidates": 40,
                "limit": 4,
            }
        }

        assert result == expected


class TestCombinePipelines:
    def test_combine_with_empty_pipeline(self):
        pipeline = []
        stage = [{"$match": {"field": "value"}}]

        combine_pipelines(pipeline, stage, "test_collection")

        assert pipeline == stage

    def test_combine_with_existing_pipeline(self):
        pipeline = [{"$search": {"index": "test"}}]
        stage = [{"$vectorSearch": {"index": "vector_test"}}]

        combine_pipelines(pipeline, stage, "test_collection")

        expected_union = {"$unionWith": {"coll": "test_collection", "pipeline": stage}}

        assert len(pipeline) == 2
        assert pipeline[1] == expected_union

    def test_combine_modifies_in_place(self):
        original_pipeline = [{"$match": {"test": True}}]
        pipeline = original_pipeline.copy()
        stage = [{"$project": {"field": 1}}]

        combine_pipelines(pipeline, stage, "collection")

        assert len(original_pipeline) == 1
        assert len(pipeline) == 2


class TestReciprocalRankStage:
    def test_basic_reciprocal_rank(self):
        result = reciprocal_rank_stage(score_field="text_score")

        expected = [
            {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
            {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
            {
                "$addFields": {
                    "docs.text_score": {
                        "$multiply": [1, {"$divide": [1.0, {"$add": ["$rank", 0, 1]}]}]
                    },
                    "docs.rank": "$rank",
                    "_id": "$docs._id",
                }
            },
            {"$replaceRoot": {"newRoot": "$docs"}},
        ]

        assert result == expected

    def test_reciprocal_rank_with_penalty(self):
        result = reciprocal_rank_stage(score_field="vector_score", penalty=60)

        add_fields_stage = result[2]["$addFields"]
        divide_expr = add_fields_stage["docs.vector_score"]["$multiply"][1]["$divide"]
        add_expr = divide_expr[1]["$add"]

        assert add_expr == ["$rank", 60, 1]

    def test_reciprocal_rank_custom_score_field(self):
        result = reciprocal_rank_stage(score_field="custom_score_field")

        add_fields_stage = result[2]["$addFields"]
        assert "docs.custom_score_field" in add_fields_stage

    def test_reciprocal_rank_with_kwargs(self):
        result = reciprocal_rank_stage(score_field="test_score", penalty=10, extra_param="ignored")

        assert len(result) == 4
        assert result[2]["$addFields"]["docs.test_score"]["$multiply"][1]["$divide"][1]["$add"] == [
            "$rank",
            10,
            1,
        ]


class TestFinalHybridStage:
    def test_basic_final_hybrid(self):
        result = final_hybrid_stage(scores_fields=["text_score", "vector_score"], limit=10)

        expected = [
            {"$group": {"_id": "$_id", "docs": {"$mergeObjects": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$docs"}},
            {
                "$set": {
                    "text_score": {"$ifNull": ["$text_score", 0]},
                    "vector_score": {"$ifNull": ["$vector_score", 0]},
                }
            },
            {"$addFields": {"score": {"$add": ["$text_score", "$vector_score"]}}},
            {"$sort": {"score": -1}},
            {"$limit": 10},
        ]

        assert result == expected

    def test_final_hybrid_single_score(self):
        result = final_hybrid_stage(scores_fields=["single_score"], limit=5)

        set_stage = result[2]["$set"]
        assert set_stage == {"single_score": {"$ifNull": ["$single_score", 0]}}

        add_fields_stage = result[3]["$addFields"]
        assert add_fields_stage == {"score": {"$add": ["$single_score"]}}

        assert result[5] == {"$limit": 5}

    def test_final_hybrid_multiple_scores(self):
        scores = ["text_score", "vector_score", "semantic_score"]
        result = final_hybrid_stage(scores_fields=scores, limit=20)

        set_stage = result[2]["$set"]
        for score in scores:
            assert score in set_stage
            assert set_stage[score] == {"$ifNull": [f"${score}", 0]}

        add_fields_stage = result[3]["$addFields"]
        expected_add = {"$add": [f"${score}" for score in scores]}
        assert add_fields_stage["score"] == expected_add

    def test_final_hybrid_with_kwargs(self):
        result = final_hybrid_stage(scores_fields=["test_score"], limit=15, extra_param="ignored")

        assert len(result) == 6
        assert result[5] == {"$limit": 15}


class TestPipelineIntegration:
    def test_text_and_vector_pipeline_components(self):
        text_pipeline = text_search_stage(
            query="machine learning", search_field="content", index_name="text_index", limit=10
        )

        vector_stage = vector_search_stage(
            query_vector=[0.1, 0.2, 0.3],
            search_field="embedding",
            index_name="vector_index",
            top_k=10,
        )

        assert isinstance(text_pipeline, list)
        assert isinstance(vector_stage, dict)
        assert "$search" in text_pipeline[0]
        assert "$vectorSearch" in vector_stage

    def test_rrf_and_final_stages_compatibility(self):
        rrf_stage = reciprocal_rank_stage(score_field="text_score")
        final_stage = final_hybrid_stage(scores_fields=["text_score", "vector_score"], limit=5)

        rrf_field_creation = rrf_stage[2]["$addFields"]
        assert "docs.text_score" in rrf_field_creation

        final_set_stage = final_stage[2]["$set"]
        assert "text_score" in final_set_stage
