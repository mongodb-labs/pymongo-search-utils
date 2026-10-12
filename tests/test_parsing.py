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

    def test_pipeline_match_with_in_operator(self):
        command = 'db.collection.aggregate([{$match: {status: {$in: ["active", "pending"]}}}])'
        result = parse_command(command)
        assert result == [{"$match": {"status": {"$in": ["active", "pending"]}}}]

    def test_invalid_command_missing_aggregate(self):
        command = "db.collection.find({x: 1})"
        with pytest.raises(ValueError, match="Could not extract aggregation pipeline"):
            parse_command(command)

    def test_invalid_pipeline_not_a_list(self):
        command = "db.collection.aggregate({$match: {x: 1}})"
        with pytest.raises(ValueError, match="Aggregation pipeline must be a list"):
            parse_command(command)

    def test_disallowed_call_rejected(self):
        command = "db.collection.aggregate([len([1, 2])])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_attribute_access_rejected(self):
        command = "db.collection.aggregate([datetime.now])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_bare_name_rejected(self):
        command = "db.collection.aggregate([some_var])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_subscript_rejected(self):
        command = "db.collection.aggregate([[1, 2][0]])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_pipeline_with_negative_number(self):
        command = "db.collection.aggregate([{$addFields: {n: -1}}])"
        result = parse_command(command)
        assert result == [{"$addFields": {"n": -1}}]

    def test_pipeline_with_float_and_bool(self):
        command = "db.collection.aggregate([{$addFields: {f: 1.5, b: True, x: None}}])"
        result = parse_command(command)
        assert result == [{"$addFields": {"f": 1.5, "b": True, "x": None}}]

    def test_pipeline_group_with_sum(self):
        command = "db.orders.aggregate([{$group: {_id: '$cust_id', total: {$sum: '$amount'}}}])"
        result = parse_command(command)
        assert result == [{"$group": {"_id": "$cust_id", "total": {"$sum": "$amount"}}}]

    def test_pipeline_sort_descending(self):
        command = "db.collection.aggregate([{$sort: {created: -1}}])"
        result = parse_command(command)
        assert result == [{"$sort": {"created": -1}}]

    def test_pipeline_project_with_exclusion(self):
        command = "db.collection.aggregate([{$project: {name: 1, address: 0, _id: 0}}])"
        result = parse_command(command)
        assert result == [{"$project": {"name": 1, "address": 0, "_id": 0}}]

    def test_pipeline_lookup(self):
        command = (
            'db.orders.aggregate([{$lookup: {from: "customers", '
            'localField: "cust_id", foreignField: "_id", as: "customer"}}])'
        )
        result = parse_command(command)
        assert result == [
            {
                "$lookup": {
                    "from": "customers",
                    "localField": "cust_id",
                    "foreignField": "_id",
                    "as": "customer",
                }
            }
        ]

    def test_pipeline_match_nested_document(self):
        command = 'db.collection.aggregate([{$match: {address: {city: "NYC", zip: "10001"}}}])'
        result = parse_command(command)
        assert result == [{"$match": {"address": {"city": "NYC", "zip": "10001"}}}]

    def test_pipeline_match_in_with_object_ids(self):
        command = (
            "db.collection.aggregate([{$match: {_id: {$in: ["
            "ObjectId('507f1f77bcf86cd799439011'), "
            "ObjectId('507f1f77bcf86cd799439012')]}}}])"
        )
        result = parse_command(command)
        assert result == [
            {
                "$match": {
                    "_id": {
                        "$in": [
                            ObjectId("507f1f77bcf86cd799439011"),
                            ObjectId("507f1f77bcf86cd799439012"),
                        ]
                    }
                }
            }
        ]

    def test_pipeline_match_date_with_gt(self):
        command = (
            'db.collection.aggregate([{$match: {created: {$gt: ISODate("2024-01-01T00:00:00Z")}}}])'
        )
        result = parse_command(command)
        expected_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert result == [{"$match": {"created": {"$gt": expected_dt}}}]

    @pytest.mark.parametrize("name", ["eval", "exec", "open", "input", "print", "sorted"])
    def test_non_whitelisted_call_rejected(self, name):
        command = f"db.collection.aggregate([{name}('x')])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_module_function_call_rejected(self):
        command = "db.collection.aggregate([os.getcwd()])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_fstring_rejected(self):
        command = "db.collection.aggregate([f'{1 + 1}'])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_comprehension_rejected(self):
        command = "db.collection.aggregate([x for x in [1]])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_starred_args_rejected(self):
        command = "db.collection.aggregate([ObjectId(*['507f1f77bcf86cd799439011'])])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_keyword_unpacking_rejected(self):
        command = "db.collection.aggregate([datetime(**{'year': 2000})])"
        with pytest.raises(ValueError, match="Failed to parse aggregation pipeline"):
            parse_command(command)

    def test_reserved_looking_dict_key_is_data(self):
        command = "db.collection.aggregate([{'$match': {'_id_extra': 1}}])"
        result = parse_command(command)
        assert result == [{"$match": {"_id_extra": 1}}]

    def test_pipeline_multi_stage(self):
        command = (
            "db.orders.aggregate(["
            '{$match: {status: "completed"}}, '
            '{$group: {_id: "$cust_id", total: {$sum: "$amount"}, count: {$sum: 1}}}, '
            "{$sort: {total: -1}}, "
            "{$limit: 10}"
            "])"
        )
        result = parse_command(command)
        assert result == [
            {"$match": {"status": "completed"}},
            {"$group": {"_id": "$cust_id", "total": {"$sum": "$amount"}, "count": {"$sum": 1}}},
            {"$sort": {"total": -1}},
            {"$limit": 10},
        ]


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
