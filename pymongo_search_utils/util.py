import logging
from typing import Any

from pymongo import AsyncMongoClient, MongoClient
from pymongo.driver_info import DriverInfo

logger = logging.getLogger(__file__)


def str_to_oid(str_repr: str) -> Any | str:
    """Attempt to cast string representation of id to MongoDB's internal BSON ObjectId.

    To be consistent with ObjectId, input must be a 24 character hex string.
    If it is not, MongoDB will happily use the string in the main _id index.
    Importantly, the str representation that comes out of MongoDB will have this form.

    Args:
        str_repr: id as string.

    Returns:
        ObjectID
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        return ObjectId(str_repr)
    except InvalidId:
        logger.debug(
            "ObjectIds must be 12-character byte or 24-character hex strings. "
            "Examples: b'heres12bytes', '6f6e6568656c6c6f68656768'"
        )
        return str_repr


def oid_to_str(oid: Any) -> str:
    """Convert MongoDB's internal BSON ObjectId into a simple str for compatibility.

    Instructive helper to show where data is coming out of MongoDB.

    Args:
        oid: bson.ObjectId

    Returns:
        24 character hex string.
    """
    return str(oid)


def append_client_metadata(client: MongoClient | AsyncMongoClient, driver_info: DriverInfo) -> None:
    # append_metadata was added in PyMongo 4.14.0, but is a valid database name on earlier versions
    if callable(client.append_metadata):
        client.append_metadata(driver_info)
