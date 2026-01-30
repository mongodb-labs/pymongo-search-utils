"""Tests for parsing utilities."""

from datetime import date, datetime, timezone

import pytest
from bson import ObjectId
from bson.binary import Binary
from bson.decimal128 import Decimal128

from pymongo_search_utils.parsing import parse_command, parse_doc_schema


class TestParseCommand:
    def test_simple_aggregation_pipeline(self):
        command = "db.collection.aggregate([{$match: {x: 1}}, {$project: {y: 1}}])"
        result = parse_command(command)
        assert result == [{"$match": {"x": 1}}, {"$project": {"y": 1}}]

    def test_simple_aggregation_pipeline_quoted(self):
        command = 'db.collection.aggregate([{"$match": {"x": 1}}, {"$project": {"y": 1}}])'
        result = parse_command(command)
        assert result == [{"$match": {"x": 1}}, {"$project": {"y": 1}}]

    def test_pipeline_missing_closing_paren(self):
        command = "db.collection.aggregate([{$match: {x: 1}}]"
        result = parse_command(command)
        assert result == [{"$match": {"x": 1}}]

    def test_pipeline_with_object_id(self):
        command = "db.collection.aggregate([{$match: {_id: ObjectId('507f1f77bcf86cd799439011')}}])"
        result = parse_command(command)
        assert result == [{"$match": {"_id": ObjectId("507f1f77bcf86cd799439011")}}]

    def test_pipeline_with_iso_date(self):
        command = 'db.collection.aggregate([{$match: {created: ISODate("2024-01-15T10:30:00Z")}}])'
        result = parse_command(command)
        expected_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert result == [{"$match": {"created": expected_dt}}]

    def test_pipeline_with_new_date(self):
        command = 'db.collection.aggregate([{$match: {created: new Date("2024-01-15T10:30:00Z")}}])'
        result = parse_command(command)
        expected_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert result == [{"$match": {"created": expected_dt}}]

    def test_pipeline_with_whitespace(self):
        command = """
            db.collection.aggregate([
                {  $match: {    status:   "active"}},
                    {   $limit: 10    }
            ])
        """
        result = parse_command(command)
        assert result == [{"$match": {"status": "active"}}, {"$limit": 10}]

    def test_invalid_command_missing_aggregate(self):
        command = "db.collection.find({x: 1})"
        with pytest.raises(ValueError, match="Could not extract aggregation pipeline"):
            parse_command(command)

    def test_invalid_pipeline_not_a_list(self):
        command = "db.collection.aggregate({$match: {x: 1}})"
        with pytest.raises(ValueError, match="Aggregation pipeline must be a list"):
            parse_command(command)


class TestParseDoc:
    def test_simple_string_field(self):
        doc = {"name": "John"}
        result = parse_doc_schema(doc, "")
        assert result == ["name: String"]

    def test_simple_int_field(self):
        doc = {"count": 42}
        result = parse_doc_schema(doc, "")
        assert result == ["count: Number"]

    def test_simple_float_field(self):
        doc = {"price": 19.99}
        result = parse_doc_schema(doc, "")
        assert result == ["price: Number"]

    def test_simple_bool_field(self):
        doc = {"active": True}
        result = parse_doc_schema(doc, "")
        assert result == ["active: Boolean"]

    def test_object_id_field(self):
        doc = {"_id": ObjectId("507f1f77bcf86cd799439011")}
        result = parse_doc_schema(doc, "")
        assert result == ["_id: ObjectId"]

    def test_date_field(self):
        doc = {"created": date(2024, 1, 15)}
        result = parse_doc_schema(doc, "")
        assert result == ["created: Date"]

    def test_datetime_field(self):
        doc = {"updated": datetime(2024, 1, 15, 10, 30, 0)}
        result = parse_doc_schema(doc, "")
        assert result == ["updated: Timestamp"]

    def test_decimal128_field(self):
        doc = {"amount": Decimal128("123.45")}
        result = parse_doc_schema(doc, "")
        assert result == ["amount: Decimal128"]

    def test_binary_field(self):
        doc = {"data": Binary(b"test")}
        result = parse_doc_schema(doc, "")
        assert result == ["data: Binary"]

    def test_nested_document(self):
        doc = {"user": {"name": "John", "age": 30}}
        result = parse_doc_schema(doc, "")
        assert "user.name: String" in result
        assert "user.age: Number" in result

    def test_deeply_nested_document(self):
        doc = {"level1": {"level2": {"level3": "value"}}}
        result = parse_doc_schema(doc, "")
        assert result == ["level1.level2.level3: String"]

    def test_empty_array(self):
        doc = {"items": []}
        result = parse_doc_schema(doc, "")
        assert result == ["items: Array"]

    def test_array_of_strings(self):
        doc = {"tags": ["a", "b", "c"]}
        result = parse_doc_schema(doc, "")
        assert result == ["tags: Array<String>"]

    def test_array_of_numbers(self):
        doc = {"scores": [1, 2, 3]}
        result = parse_doc_schema(doc, "")
        assert result == ["scores: Array<Number>"]

    def test_array_of_documents(self):
        doc = {"items": [{"name": "item1", "qty": 5}]}
        result = parse_doc_schema(doc, "")
        assert "items[].name: String" in result
        assert "items[].qty: Number" in result

    def test_with_prefix(self):
        doc = {"name": "John"}
        result = parse_doc_schema(doc, "root")
        assert result == ["root.name: String"]

    def test_empty_document_with_prefix(self):
        doc = {}
        result = parse_doc_schema(doc, "empty")
        assert result == ["empty: Document"]

    def test_multiple_fields(self):
        doc = {"name": "John", "age": 30, "active": True}
        result = parse_doc_schema(doc, "")
        assert "name: String" in result
        assert "age: Number" in result
        assert "active: Boolean" in result

    def test_array_of_unknown_type(self):
        # Using a type not in _BSON_LOOKUP
        doc = {"items": [object()]}
        result = parse_doc_schema(doc, "")
        assert result == ["items: Array"]
