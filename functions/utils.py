# functions/utils.py
"""
Utility functions for the Firebase Cloud Functions backend.

This module provides a collection of helper functions for common tasks such as
text processing, file hashing, generating Firestore/Storage paths, vector math,
and Firestore data manipulation (e.g., recursive deletion of collections).
"""
import re
import hashlib
from typing import Any, IO, TYPE_CHECKING

import numpy as np
from firebase_admin import firestore # For CollectionReference type hint

# For type checking FirestoreClient without circular imports during runtime
if TYPE_CHECKING:
    from firebase_admin.firestore import Client as FirestoreClient

# --- Text Processing ---
def clean_text(text: str) -> str:
    """
    Cleans a given text string by performing several operations:
    1. Replaces multiple whitespace characters with a single space.
    2. Replaces newline characters with a space.
    3. Converts the text to lowercase.
    4. Removes leading and trailing whitespace.

    Args:
        text: The input string to be cleaned.

    Returns:
        The cleaned string.
    """
    text = re.sub(r'\s+', ' ', text) 
    text = text.replace('\n', ' ')
    text = text.lower()
    return text.strip()

# --- File Handling ---
def get_file_hash(file_stream: IO[bytes]) -> str:
    """
    Calculates the MD5 hash of a file-like object (stream).

    The function reads the stream in chunks to handle potentially large files
    efficiently. After calculating the hash, it resets the stream's read
    pointer to the beginning (0).

    Args:
        file_stream: An IO[bytes] object (e.g., a file opened in binary mode,
                     or a SpooledTemporaryFile from Flask/Werkzeug) from which
                     to read the data. The stream must support `seek()` and `read()`.

    Returns:
        A hexadecimal string representing the MD5 hash of the file's content.
    """
    original_position = file_stream.tell() # Save original position if seekable
    file_stream.seek(0)
    hasher = hashlib.md5()
    buf = file_stream.read(65536) # Read in 64k chunks
    while len(buf) > 0:
        hasher.update(buf)
        buf = file_stream.read(65536)
    file_stream.seek(original_position) # Reset stream position to where it was
    return hasher.hexdigest()

# --- Firestore Path Generation ---
def get_user_storage_path(uid: str) -> str:
    """
    Generates the Cloud Storage path prefix for a user's uploaded documents.

    This path is typically used to organize files in a user-specific directory
    within a Firebase Storage bucket.

    Args:
        uid: The unique user ID.

    Returns:
        A string representing the storage path prefix (e.g., "user_uploads/USER_ID/docs/").
    """
    return f"user_uploads/{uid}/docs/"

def get_user_document_chunks_collection_path(
    db: 'FirestoreClient', 
    uid: str, 
    document_hash: str
) -> firestore.CollectionReference:
    """
    Generates the Firestore CollectionReference for storing chunks of a specific document
    belonging to a user.

    The path structure is typically: users/{uid}/processed_documents/{document_hash}/chunks

    Args:
        db: The Firestore client instance (e.g., `firestore.client()`).
        uid: The unique user ID.
        document_hash: The MD5 hash of the document, used as its unique identifier.

    Returns:
        A `firebase_admin.firestore.CollectionReference` pointing to the document's chunks.
    """
    return db.collection('users').document(uid).collection('processed_documents').document(document_hash).collection('chunks')

def get_user_processed_files_metadata_collection_path(
    db: 'FirestoreClient', 
    uid: str
) -> firestore.CollectionReference:
    """
    Generates the Firestore CollectionReference for storing metadata of all processed files
    for a specific user.

    The path structure is typically: users/{uid}/processed_files_metadata

    Args:
        db: The Firestore client instance (e.g., `firestore.client()`).
        uid: The unique user ID.

    Returns:
        A `firebase_admin.firestore.CollectionReference` pointing to the user's file metadata.
    """
    return db.collection('users').document(uid).collection('processed_files_metadata')

# --- Vector Operations ---
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two NumPy vectors.

    Cosine similarity measures the cosine of the angle between two non-zero vectors,
    providing a measure of their orientation similarity. A value of 1 means identical
    orientation, 0 means orthogonal, and -1 means opposite orientation.

    Args:
        v1: The first NumPy array (vector).
        v2: The second NumPy array (vector). Both v1 and v2 must have the same dimensions.

    Returns:
        A float representing the cosine similarity score. Returns 0.0 if either
        vector has a zero norm to prevent division by zero errors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: 
        return 0.0 # Or raise an error, depending on desired handling of zero vectors
    return float(dot_product / (norm_v1 * norm_v2)) # Ensure float return

# --- Firestore Data Deletion ---
def delete_collection_recursively(
    db: 'FirestoreClient', 
    coll_ref: firestore.CollectionReference, 
    batch_size: int
) -> int:
    """
    Deletes all documents in a Firestore collection and, importantly, any subcollections
    recursively.

    This function operates by fetching documents in batches and deleting them.
    If a document in the 'processed_documents' collection (identified by its parent's ID)
    is encountered, this function will recursively call itself to delete the
    associated 'chunks' subcollection before deleting the document itself.

    Args:
        db: The Firestore client instance.
        coll_ref: The `CollectionReference` of the collection to be deleted.
        batch_size: The number of documents to retrieve and delete in each batch.
                    A smaller batch size is safer but slower. Max 500 for Firestore.

    Returns:
        An integer representing the total number of documents deleted from the
        specified collection and its subcollections.
    
    Note:
        This function can be resource-intensive for very large collections.
        Consider using Firebase CLI tools or dedicated solutions for massive deletions if needed.
    """
    docs_stream = coll_ref.limit(batch_size).stream()
    deleted_count = 0
    
    # Keep track of documents in the current stream to process
    # This list will be repopulated if a batch is committed and more documents might exist.
    current_docs_to_process = list(docs_stream)

    while current_docs_to_process:
        batch_ops = db.batch()
        num_in_batch = 0

        for doc_snap in current_docs_to_process:
            # Check for 'chunks' subcollection in 'processed_documents/{doc_hash}'
            # The path is users/{uid}/processed_documents/{doc_hash}
            # So, doc_snap.reference.parent is 'processed_documents' collection
            # doc_snap.reference.parent.parent is '{uid}' document
            # doc_snap.reference.parent.parent.parent is 'users' collection
            parent_col = doc_snap.reference.parent
            if parent_col.id == "processed_documents" and \
               parent_col.parent is not None and \
               parent_col.parent.parent is not None and \
               parent_col.parent.parent.id == "users":
                
                chunks_subcollection_ref = doc_snap.reference.collection("chunks")
                # Recursively delete the 'chunks' subcollection first
                delete_collection_recursively(db, chunks_subcollection_ref, batch_size)

            batch_ops.delete(doc_snap.reference)
            num_in_batch += 1
        
        if num_in_batch > 0:
            batch_ops.commit()
            deleted_count += num_in_batch
            print(f"INFO (utils.py): Deleted batch of {num_in_batch} docs from {coll_ref.path}. Total for this op: {deleted_count}")
        
        # If we processed a full batch, there might be more documents.
        # Otherwise (less than batch_size), we've processed all remaining documents.
        if num_in_batch < batch_size:
            break 
        else:
            # Re-fetch the next batch of documents
            docs_stream = coll_ref.limit(batch_size).stream()
            current_docs_to_process = list(docs_stream)
            if not current_docs_to_process: # No more documents
                break
    
    return deleted_count
