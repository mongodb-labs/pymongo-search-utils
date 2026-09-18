# Changelog

______________________________________________________________________

## Changes in version 0.3.1 (XXXX/XX/XX)

- Add `wait_for_fulltext_docs_in_index`, which waits for a fulltext index to catch up
  with documents inserted after the index was created.
- Fix `drop_vector_search_index`. It was waiting for the collection to hold zero search
  indexes rather than for the named index to be gone. It previously timed out on any
  collection that keeps another index.
- Rename `drop_vector_search_index` to `drop_search_index`. It drops any search index by
  name and never did anything vector-specific. The old name remains as a deprecated alias.
- `wait_for_docs_in_index` now selects the vector field by type rather than assuming it is
  first in the index definition, raises `TimeoutError` instead of returning `False`, and
  raises `ValueError` when given a fulltext index. Its docstring documented a parameter
  that does not exist.

## Changes in version 0.3.0 (2026/2/3)

- Add utilities for MongoDB schema and LLM text to command parsing.

## Changes in version 0.2.1 (2026/1/14)

- Add type support (`py.typed` file).

## Changes in version 0.2.0 (2026/1/14)

- Add support for auto-embeddings (`autoembedding_vector_search_stage`).

## Changes in version 0.1.0 (2025/11/24)

- Initial release.
