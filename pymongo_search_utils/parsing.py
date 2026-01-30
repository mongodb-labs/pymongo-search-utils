"""Parsing utilities and helpers."""

import re
from datetime import date, datetime, timezone
from typing import Any

from bson import ObjectId
from bson.binary import Binary
from bson.decimal128 import Decimal128

_BSON_LOOKUP = {
    str: "String",
    int: "Number",
    float: "Number",
    bool: "Boolean",
    ObjectId: "ObjectId",
    date: "Date",
    datetime: "Timestamp",
    None: "Null",
    Decimal128: "Decimal128",
    Binary: "Binary",
}


def parse_command(command: str) -> Any:
    """
    Extracts and parses the aggregation pipeline from a JavaScript-style MongoDB command.
    Handles ObjectId(), ISODate(), new Date() and converts them into Python constructs.
    """
    command = re.sub(r"\s+", " ", command.strip())
    if command.endswith("]"):
        command += ")"

    try:
        agg_str = command.split(".aggregate(", 1)[1].rsplit(")", 1)[0]
    except Exception as e:
        raise ValueError(f"Could not extract aggregation pipeline: {e}") from e

    # Convert JavaScript-style constructs to Python syntax
    agg_str = _convert_mongo_js_to_python(agg_str)

    try:
        eval_globals = {
            "ObjectId": ObjectId,
            "datetime": datetime,
            "timezone": timezone,
        }
        agg_pipeline = eval(agg_str, eval_globals)
        if not isinstance(agg_pipeline, list):
            raise ValueError("Aggregation pipeline must be a list.")
        return agg_pipeline
    except Exception as e:
        raise ValueError(f"Failed to parse aggregation pipeline: {e}") from e


def parse_doc(doc: dict[str, Any], prefix: str) -> list[str]:
    sub_schema = []
    for key, value in doc.items():
        if prefix:
            full_key = f"{prefix}.{key}"
        else:
            full_key = key
        if isinstance(value, dict):
            sub_schema.extend(parse_doc(value, full_key))
        elif isinstance(value, list):
            if not len(value):
                sub_schema.append(f"{full_key}: Array")
            elif isinstance(value[0], dict):
                sub_schema.extend(parse_doc(value[0], f"{full_key}[]"))
            else:
                if type(value[0]) in _BSON_LOOKUP:
                    type_name = _BSON_LOOKUP[type(value[0])]
                    sub_schema.append(f"{full_key}: Array<{type_name}>")
                else:
                    sub_schema.append(f"{full_key}: Array")
        elif type(value) in _BSON_LOOKUP:
            type_name = _BSON_LOOKUP[type(value)]
            sub_schema.append(f"{full_key}: {type_name}")
    if not sub_schema:
        sub_schema.append(f"{prefix}: Document")
    return sub_schema


def _convert_mongo_js_to_python(code: str) -> str:
    """Convert JavaScript-style MongoDB syntax into Python-safe code."""

    def _handle_iso_date(match: Any) -> str:
        date_str = match.group(1)
        if not date_str:
            raise ValueError("ISODate must contain a date string.")
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return (
            f"datetime({dt.year}, "
            f"{dt.month}, "
            f"{dt.day}, "
            f"{dt.hour}, "
            f"{dt.minute}, "
            f"{dt.second}, tzinfo=timezone.utc)"
        )

    def _handle_new_date(match: Any) -> str:
        date_str = match.group(1)
        if not date_str:
            raise ValueError(
                "new Date() without arguments is not allowed. Please pass an explicit date string."
            )
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return (
            f"datetime({dt.year},"
            f" {dt.month},"
            f" {dt.day},"
            f" {dt.hour},"
            f" {dt.minute},"
            f" {dt.second}, tzinfo=timezone.utc)"
        )

    def _handle_object_id(match: Any) -> str:
        oid_str = match.group(1)
        if not oid_str:
            raise ValueError("ObjectId must contain a value.")
        return f"ObjectId('{oid_str}')"

    def _handle_id_key(match: Any) -> str:
        return f'"{match.group(1)}"'

    patterns = [
        (r'ISODate\(\s*["\']([^"\']*)["\']\s*\)', _handle_iso_date),
        (r'new\s+Date\(\s*["\']([^"\']*)["\']\s*\)', _handle_new_date),
        (r'ObjectId\(\s*["\']([^"\']*)["\']\s*\)', _handle_object_id),
        (r'ObjectId\(\s*["\']([^"\']*)["\']\s*\)', _handle_object_id),
        (r'(?<!["\'])\b(_id)\b(?!["\'])', _handle_id_key),
    ]

    for pattern, replacer in patterns:
        code = re.sub(pattern, replacer, code)

    return code
