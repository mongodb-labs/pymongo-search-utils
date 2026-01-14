from ._version import __version__
from .index import (
    create_fulltext_search_index,
    create_vector_search_index,
    drop_vector_search_index,
    update_vector_search_index,
)
from .operation import bulk_embed_and_insert_texts
from .pipeline import (
    autoembedding_vector_search_stage,
    combine_pipelines,
    final_hybrid_stage,
    reciprocal_rank_stage,
    text_search_stage,
    vector_search_stage,
)
from .util import append_client_metadata

__all__ = [
    "__version__",
    "create_vector_search_index",
    "drop_vector_search_index",
    "update_vector_search_index",
    "create_fulltext_search_index",
    "text_search_stage",
    "vector_search_stage",
    "autoembedding_vector_search_stage",
    "combine_pipelines",
    "reciprocal_rank_stage",
    "final_hybrid_stage",
    "bulk_embed_and_insert_texts",
    "append_client_metadata",
]
