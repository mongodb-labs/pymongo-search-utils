import os
from collections.abc import Generator

import pytest
from pymongo import MongoClient
from pymongo.errors import OperationFailure

# Raised by a deployment that has no mongot: "Using Atlas Search Database
# Commands and the $listSearchIndexes aggregation stage requires additional
# configuration. Please connect to Atlas or an AtlasCLI local deployment."
SEARCH_NOT_ENABLED = 31082


@pytest.fixture(scope="session")
def dbname() -> str:
    return "pymongo_search_utils_test"


@pytest.fixture(scope="session")
def client() -> Generator[MongoClient, None, None]:
    conn_str = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017?directConnection=true")
    client = MongoClient(conn_str)
    yield client
    client.close()


@pytest.fixture(scope="session")
def search_available(client: MongoClient, dbname: str) -> bool:
    try:
        client[dbname]["_search_capability_probe"].list_search_indexes().to_list()
    except OperationFailure as exc:
        if exc.code == SEARCH_NOT_ENABLED:
            return False
        raise
    return True


@pytest.fixture
def requires_search(search_available: bool) -> None:
    if not search_available:
        pytest.skip("Deployment does not have Atlas Search enabled.")
