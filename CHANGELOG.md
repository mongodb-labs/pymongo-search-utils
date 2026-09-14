# Changelog

______________________________________________________________________

## Changes in version 0.3.1 (XXXX/XX/XX)

- Add `wait_for_fulltext_docs_in_index`, which waits for a fulltext index to catch up
  with documents inserted after the index was created.
- Fix `drop_vector_search_index`. It was waiting for the collection to hold zero search
  indexes rather than for the named index to be gone. It previously timed out on any
  collection that keeps another index.

## Changes in version 0.3.0 (2026/2/3)

- Add utilities for MongoDB schema and LLM text to command parsing.

## Changes in version 0.2.1 (2026/1/14)

- Add type support (`py.typed` file).

## Changes in version 0.2.0 (2026/1/14)

- Add support for auto-embeddings (`autoembedding_vector_search_stage`).

## Changes in version 0.1.0 (2025/11/24)

- Initial release.
