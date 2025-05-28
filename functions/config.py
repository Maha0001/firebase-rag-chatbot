# functions/config.py
"""
Configuration constants for the Firebase Cloud Functions backend.

This module stores various settings and parameters used throughout the application,
such as AI model names, Firestore batch sizes, text processing parameters,
and other configurable values.
"""
from typing import Final

EMBEDDING_MODEL_NAME: Final[str] = "models/text-embedding-004"
"""The name of the text embedding model to be used."""

LLM_MODEL_NAME: Final[str] = "gemini-2.5-pro-preview-05-06"
"""The name of the large language model to be used for chat generation."""

DEFAULT_FIRESTORE_BATCH_SIZE: Final[int] = 100
"""
Default batch size for Firestore operations, particularly for recursive deletions.
This helps manage resource consumption and avoid exceeding Firestore limits.
"""

CHUNK_SIZE: Final[int] = 1000
"""
The target size for text chunks when splitting documents for embedding.
Measured in characters.
"""

CHUNK_OVERLAP: Final[int] = 200
"""
The number of characters of overlap between consecutive text chunks.
This helps maintain context across chunk boundaries.
"""

CHAT_SIMILARITY_TOP_K: Final[int] = 3
"""
The number of top similar document chunks to retrieve for providing context to the LLM in the chat feature.
"""

FIRESTORE_BATCH_MAX_WRITES: Final[int] = 490
"""
Maximum number of write operations (set, update, delete) to include in a single Firestore batch commit.
Firestore has a hard limit of 500 operations per batch. This provides a small safety margin.
Relevant for operations like uploading PDF chunks.
"""

MAX_CHUNKS_FOR_SIMILARITY_SCORING: Final[int] = 5000
"""
Maximum number of chunks to load and score for similarity in the chat function.
Helps prevent performance issues for users with very large datasets.
A trade-off: might miss relevant older chunks if the limit is hit, especially
if documents are not processed in a relevance-based order.
"""
