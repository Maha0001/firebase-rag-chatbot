# functions/routes.py
"""
Flask Blueprint and route handlers for the API endpoints.

This module defines the main API routes for the application, including
PDF upload, chat interaction, and user data reset. All routes are protected
by the `@require_auth` decorator, which ensures that the user is authenticated
and injects the user ID (UID) into the route handler.
"""
import os
import shutil # For /tmp cleanup
from typing import Any, Dict, List, Tuple, Union, IO

from flask import Blueprint, jsonify, request
from flask import Response as FlaskResponse # For type hinting Flask responses
from firebase_admin import firestore # For SERVER_TIMESTAMP
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage # For type hinting request.files['file']
import numpy as np

# Langchain and related imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument # Alias to avoid conflict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Project-specific imports
from .auth_utils import require_auth
from .utils import (
    clean_text, 
    get_file_hash, 
    get_user_storage_path,
    get_user_document_chunks_collection_path,
    get_user_processed_files_metadata_collection_path,
    cosine_similarity,
    delete_collection_recursively 
)
from .main import db, bucket, get_embedding_model, get_llm 
from .config import (
    CHAT_SIMILARITY_TOP_K, 
    FIRESTORE_BATCH_MAX_WRITES, 
    DEFAULT_FIRESTORE_BATCH_SIZE, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    MAX_CHUNKS_FOR_SIMILARITY_SCORING # New import
)

api_bp = Blueprint('api', __name__)
"""
Flask Blueprint for the API routes.
All routes defined in this blueprint will be registered with the main Flask app.
"""

# Define a more specific type for Flask JSON responses for clarity
JsonResponse = Tuple[FlaskResponse, int]

@api_bp.route('/upload_pdf', methods=['POST'])
@require_auth
def http_upload_pdf(uid: str) -> JsonResponse:
    """
    Handles PDF file uploads, processes them, generates embeddings, and stores them.

    The user must be authenticated. The PDF file is expected in the 'file' part of
    a multipart/form-data request.

    Args:
        uid: The authenticated user's ID, injected by the `@require_auth` decorator.

    Returns:
        A Flask JSON response tuple `(flask.Response, int)`:
        - On success (200 or 201): A message indicating success and the document hash.
        - On error (400, 500, 503): An error message.
    """
    current_embeddings_model = get_embedding_model()
    if not current_embeddings_model:
        return jsonify({"error": "Embedding service not available. Check API key or server logs."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file_from_request: Union[FileStorage, Any] = request.files['file'] # Type hint for clarity
    if not file_from_request or not file_from_request.filename or not file_from_request.filename.lower().endswith('.pdf'):
        return jsonify({"error": "No selected file, filename is empty, or file is not a PDF."}), 400

    original_filename = secure_filename(file_from_request.filename)
    
    # file_stream is IO[bytes] as per Werkzeug's FileStorage.stream
    file_stream: IO[bytes] = file_from_request.stream 
    document_hash = get_file_hash(file_stream)

    processed_files_metadata_col = get_user_processed_files_metadata_collection_path(db, uid)
    doc_meta_ref = processed_files_metadata_col.document(document_hash)

    if doc_meta_ref.get().exists:
        return jsonify({"message": f"File '{original_filename}' (hash: {document_hash}) already processed."}), 200

    storage_path_prefix = get_user_storage_path(uid)
    blob_name = f"{document_hash}_{original_filename}"
    blob_path = f"{storage_path_prefix}{blob_name}"
    
    gcs_blob = bucket.blob(blob_path)
    file_stream.seek(0) # Reset stream after hashing
    gcs_blob.upload_from_file(file_stream, content_type='application/pdf')
    print(f"INFO (routes.py): Uploaded '{original_filename}' to '{blob_path}' for user '{uid}'")

    temp_pdf_path = f"/tmp/{blob_name}"
    file_stream.seek(0)
    try:
        with open(temp_pdf_path, 'wb') as f_temp:
            shutil.copyfileobj(file_stream, f_temp)
    finally:
        file_stream.close() # Close the stream from Flask request

    chunk_count_successfully_embedded = 0
    all_langchain_chunks: List[LangchainDocument] = []
    try:
        loader = PyPDFLoader(temp_pdf_path)
        raw_documents: List[LangchainDocument] = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP, 
            separators=["\n\n", "\n", ". ", " ", ""], 
            keep_separator=False
        )
        
        for page_doc in raw_documents:
            # Ensure page_content is not None before processing
            page_content_str = page_doc.page_content if page_doc.page_content is not None else ""
            page_metadata = page_doc.metadata if page_doc.metadata is not None else {}
            page_chunks = text_splitter.create_documents([page_content_str], metadatas=[page_metadata])
            all_langchain_chunks.extend(page_chunks)

        chunks_collection_ref = get_user_document_chunks_collection_path(db, uid, document_hash)
        
        current_batch_ops = db.batch()
        current_batch_write_count = 0
        
        for i, chunk_doc_obj in enumerate(all_langchain_chunks):
            text_content_for_embedding = clean_text(chunk_doc_obj.page_content)
            if not text_content_for_embedding: 
                continue

            try:
                embedding_vector = current_embeddings_model.embed_query(text_content_for_embedding)
            except Exception as e:
                print(f"ERROR (routes.py): Embedding failed for chunk {i} of '{original_filename}': {e}")
                continue 

            chunk_firestore_doc_ref = chunks_collection_ref.document(f"chunk_{i:04d}")
            current_batch_ops.set(chunk_firestore_doc_ref, {
                "original_text": chunk_doc_obj.page_content,
                "text_for_retrieval": text_content_for_embedding,
                "embedding": embedding_vector,
                "document_hash": document_hash,
                "original_filename": original_filename,
                "chunk_index": i,
                "source_metadata": chunk_doc_obj.metadata,
                "created_at": firestore.SERVER_TIMESTAMP
            })
            chunk_count_successfully_embedded += 1
            current_batch_write_count += 1
            
            if current_batch_write_count >= FIRESTORE_BATCH_MAX_WRITES:
                current_batch_ops.commit()
                print(f"INFO (routes.py): Committed batch of {current_batch_write_count} chunks for '{original_filename}'")
                current_batch_ops = db.batch()
                current_batch_write_count = 0
        
        if current_batch_write_count > 0:
            current_batch_ops.commit()
            print(f"INFO (routes.py): Committed final batch of {current_batch_write_count} chunks for '{original_filename}'")
        
        print(f"INFO (routes.py): Stored {chunk_count_successfully_embedded} embeddable chunks for '{original_filename}'")

        status = "processed" if chunk_count_successfully_embedded > 0 else "processing_failed_no_chunks_embedded"
        doc_meta_ref.set({
            "original_filename": original_filename, "storage_path": blob_path,
            "document_hash": document_hash, "total_chunks_processed": len(all_langchain_chunks),
            "chunks_successfully_embedded": chunk_count_successfully_embedded, "status": status,
            "uploaded_at": firestore.SERVER_TIMESTAMP, "last_processed_at": firestore.SERVER_TIMESTAMP
        })
        
    except Exception as e:
        print(f"ERROR (routes.py): PDF processing failed for '{original_filename}' (user '{uid}'): {e}")
        import traceback
        traceback.print_exc()
        doc_meta_ref.set({
            "original_filename": original_filename, "document_hash": document_hash,
            "status": "processing_failed", "error": str(e),
            "uploaded_at": firestore.SERVER_TIMESTAMP,
            "last_processed_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    if chunk_count_successfully_embedded == 0 and len(all_langchain_chunks) > 0:
        error_msg = "PDF processed, but no text chunks could be embedded. Document might be empty or unparseable."
        return jsonify({"error": error_msg, "document_hash": document_hash}), 500
    
    return jsonify({
        "message": f"File '{original_filename}' processed successfully. Stored {chunk_count_successfully_embedded} embeddable chunks.",
        "document_hash": document_hash
    }), 201


@api_bp.route('/chat', methods=['POST'])
@require_auth
def http_chat(uid: str) -> JsonResponse:
    """
    Handles chat requests from authenticated users.

    It expects a JSON payload with a 'query' (the user's question) and
    optionally 'chat_history' (a list of previous messages).
    It retrieves relevant document chunks based on the query, constructs a
    context, and uses an LLM to generate a response.

    Args:
        uid: The authenticated user's ID, injected by `@require_auth`.

    Returns:
        A Flask JSON response tuple `(flask.Response, int)`:
        - On success (200): The LLM's generated answer.
        - On error (400, 500, 503): An error message.
    """
    current_embeddings_model = get_embedding_model()
    current_llm = get_llm()
    if not current_embeddings_model or not current_llm:
        return jsonify({"error": "AI services not available. Check API key or server logs."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received in request body."}), 400
        
    query_text: Optional[str] = data.get('query')
    frontend_chat_history: List[Dict[str, str]] = data.get('chat_history', []) # Ensure it defaults to empty list

    if not query_text:
        return jsonify({"error": "No 'query' provided in the request payload."}), 400

    try:
        query_embedding_list = current_embeddings_model.embed_query(query_text)
        np_query_embedding = np.array(query_embedding_list)

        processed_files_col = get_user_processed_files_metadata_collection_path(db, uid)
        user_doc_metas_query = processed_files_col.where("status", "==", "processed").stream()
        
        all_user_chunks_for_similarity: List[Dict[str, Any]] = []
        for doc_meta_snap in user_doc_metas_query:
            doc_hash = doc_meta_snap.id
            chunks_col_ref = get_user_document_chunks_collection_path(db, uid, doc_hash)
            chunk_docs_stream = chunks_col_ref.stream() 
            for chunk_doc_snap in chunk_docs_stream:
                chunk_data = chunk_doc_snap.to_dict()
                if chunk_data and 'embedding' in chunk_data and 'text_for_retrieval' in chunk_data:
                    all_user_chunks_for_similarity.append({
                        'id': chunk_doc_snap.id, 
                        'text': chunk_data['text_for_retrieval'],
                        'embedding': np.array(chunk_data['embedding']), # Ensure it's a NumPy array
                        'original_filename': chunk_data.get('original_filename', 'N/A'),
                        'chunk_index': chunk_data.get('chunk_index', -1),
                        'source_metadata': chunk_data.get('source_metadata', {})
                    })
        
        if not all_user_chunks_for_similarity:
            return jsonify({"answer": "I don't have any searchable documents for you yet. Please upload some PDFs first."}), 200

        similarities: List[Tuple[float, Dict[str, Any]]] = []
        for chunk_info in all_user_chunks_for_similarity:
            sim = cosine_similarity(np_query_embedding, chunk_info['embedding'])
            similarities.append((sim, chunk_info))

        similarities.sort(key=lambda x: x[0], reverse=True)
        
        retrieved_langchain_docs: List[LangchainDocument] = []
        added_texts_for_context: set[str] = set()
        for sim_score, chunk_info in similarities:
            if len(retrieved_langchain_docs) >= CHAT_SIMILARITY_TOP_K: 
                break
            chunk_text = chunk_info.get('text')
            if chunk_text and chunk_text not in added_texts_for_context:
                doc_metadata = {
                    "source_filename": chunk_info.get('original_filename', 'Unknown'), 
                    "chunk_id": chunk_info.get('id', 'Unknown'),
                    "original_chunk_index": chunk_info.get('chunk_index', -1),
                    "page_number": chunk_info.get('source_metadata', {}).get('page', 'N/A'),
                    "similarity_score": float(sim_score) 
                }
                retrieved_langchain_docs.append(LangchainDocument(page_content=chunk_text, metadata=doc_metadata))
                added_texts_for_context.add(chunk_text)

        if not retrieved_langchain_docs:
             return jsonify({"answer": "I found documents, but couldn't find information specifically relevant to your query in them."}), 200

        context_str = "\n\n---\n\n".join(
            [f"Source: {doc.metadata.get('source_filename', 'Unknown')} "
             f"(Page {doc.metadata.get('page_number', 'N/A')}, Chunk approx. {doc.metadata.get('original_chunk_index', 'N/A')})\n"
             f"Content: {doc.page_content}" for doc in retrieved_langchain_docs]
        )
        
        chat_history_for_prompt_list: List[str] = []
        for entry in frontend_chat_history:
            role = "Human" if entry.get("role") == "user" else "Assistant"
            chat_history_for_prompt_list.append(f"{role}: {entry.get('content', '')}") # Ensure content exists
        chat_history_str_for_prompt = "\n".join(chat_history_for_prompt_list)
        
        template = """You are a helpful AI assistant answering questions based on the provided context from documents and the ongoing chat history.
Your goal is to be informative and rely strictly on the provided document context.
If the answer cannot be found in the context, explicitly state that the information is not available in the provided documents.
Do not make up answers or use external knowledge.

Chat History:
{chat_history}

Provided Context from Documents:
{context}

User's Question: {question}

Helpful Answer (based ONLY on the chat history and provided context):"""
        
        prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)
        chain = LLMChain(llm=current_llm, prompt=prompt)
        
        response = chain.invoke({
            "chat_history": chat_history_str_for_prompt, 
            "context": context_str, 
            "question": query_text
        })
        answer = response.get("text", "I apologize, but I encountered an issue generating a response.") # Default if 'text' key is missing
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"ERROR (routes.py): Chat processing error for user '{uid}': {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred during chat processing: {str(e)}"}), 500


@api_bp.route('/reset_user_data', methods=['POST'])
@require_auth
def http_reset_user_data(uid: str) -> JsonResponse:
    """
    Resets all data for the authenticated user.

    This includes deleting all documents uploaded by the user from Cloud Storage,
    and all associated metadata and processed chunks from Firestore.
    This is a destructive operation.

    Args:
        uid: The authenticated user's ID, injected by `@require_auth`.

    Returns:
        A Flask JSON response tuple `(flask.Response, int)`:
        - On success (200): A message confirming data reset.
        - On error (500): An error message.
    """
    try:
        print(f"INFO (routes.py): Initiating data reset for user '{uid}'...")
        storage_path_prefix = get_user_storage_path(uid)
        
        blobs_to_delete = list(bucket.list_blobs(prefix=storage_path_prefix))
        for blob_item in blobs_to_delete:
            blob_item.delete()
            print(f"INFO (routes.py): Deleted '{blob_item.name}' from Storage for user '{uid}'.")
        if not blobs_to_delete:
            print(f"INFO (routes.py): No files found in Storage for user '{uid}' at prefix '{storage_path_prefix}'.")

        user_ref = db.collection('users').document(uid)
        
        processed_files_meta_col = get_user_processed_files_metadata_collection_path(db, uid)
        print(f"INFO (routes.py): Deleting 'processed_files_metadata' for user '{uid}'...")
        delete_collection_recursively(db, processed_files_meta_col, DEFAULT_FIRESTORE_BATCH_SIZE)

        processed_documents_main_col_ref = user_ref.collection('processed_documents')
        print(f"INFO (routes.py): Deleting 'processed_documents' (and their 'chunks') for user '{uid}'...")
        delete_collection_recursively(db, processed_documents_main_col_ref, DEFAULT_FIRESTORE_BATCH_SIZE)

        print(f"INFO (routes.py): Data reset completed successfully for user '{uid}'.")
        return jsonify({"message": "User data has been reset successfully."}), 200
    except Exception as e:
        print(f"ERROR (routes.py): Data reset failed for user '{uid}': {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to reset user data due to an internal error: {str(e)}"}), 500
