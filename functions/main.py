# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_admin import initialize_app

import os
import re
# import json # Not directly used now, but often useful
import hashlib
# from pathlib import Path # Not strictly needed in this version
import shutil # For /tmp cleanup

from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
from firebase_admin import initialize_app, auth, firestore, storage
import firebase_admin # Explicit import for initialize_app
from werkzeug.utils import secure_filename
import numpy as np

from langchain_community.document_loaders import PyPDFLoader # CORRECTED IMPORT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document # For creating Document objects
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables (e.g., from .env or .runtimeconfig.json for emulators)
# This should be one of the first things.
_ = load_dotenv(find_dotenv())

# Initialize Firebase Admin SDK
try:
    firebase_admin.initialize_app()
except ValueError:
    pass # App already initialized

db = firestore.client()
bucket = storage.bucket() # Default bucket

# --- Global Variables for Lazy Loaded Clients ---
# These will be initialized on first use within a request.
_embeddings_model_client = None
_llm_client = None
_google_api_key_fetched = None # To store the fetched API key

def _get_google_api_key():
    """Fetches the Google API key from environment once."""
    global _google_api_key_fetched
    if _google_api_key_fetched is None: # Fetch only once
        _google_api_key_fetched = os.environ.get("GOOGLE_API_KEY")
        if not _google_api_key_fetched:
            print("CRITICAL (main.py): GOOGLE_API_KEY not found in environment during first fetch.")
        else:
            print("INFO (main.py): GOOGLE_API_KEY fetched successfully.")
    return _google_api_key_fetched

def get_embedding_model():
    """Lazily initializes and returns the embedding model client."""
    global _embeddings_model_client
    api_key = _get_google_api_key()
    if _embeddings_model_client is None:
        if api_key:
            try:
                print("INFO (main.py): Initializing GoogleGenerativeAIEmbeddings on demand...")
                _embeddings_model_client = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=api_key
                )
                print("INFO (main.py): GoogleGenerativeAIEmbeddings initialized.")
            except Exception as e:
                print(f"ERROR (main.py): Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
                _embeddings_model_client = None # Ensure it remains None on failure
        else:
            print("ERROR (main.py): Cannot initialize embedding model, GOOGLE_API_KEY is missing.")
            _embeddings_model_client = None
    return _embeddings_model_client

def get_llm():
    """Lazily initializes and returns the LLM client."""
    global _llm_client
    api_key = _get_google_api_key()
    if _llm_client is None:
        if api_key:
            try:
                print("INFO (main.py): Initializing ChatGoogleGenerativeAI on demand...")
                _llm_client = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro-preview-05-06",
                    google_api_key=api_key
                )
                print("INFO (main.py): ChatGoogleGenerativeAI initialized.")
            except Exception as e:
                print(f"ERROR (main.py): Failed to initialize ChatGoogleGenerativeAI: {e}")
                _llm_client = None # Ensure it remains None on failure
        else:
            print("ERROR (main.py): Cannot initialize LLM, GOOGLE_API_KEY is missing.")
            _llm_client = None
    return _llm_client

# --- Helper Functions (clean_text, get_file_hash, etc. - same as before) ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text) 
    text = text.replace('\n', ' ')
    text = text.lower()
    return text.strip()

def get_file_hash(file_stream):
    file_stream.seek(0)
    hasher = hashlib.md5()
    buf = file_stream.read(65536)
    while len(buf) > 0:
        hasher.update(buf)
        buf = file_stream.read(65536)
    file_stream.seek(0)
    return hasher.hexdigest()

def get_user_id_from_request(http_request):
    auth_header = http_request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None, ('Missing or malformed authorization token', 401)
    id_token = auth_header.split('Bearer ')[-1]
    if not id_token:
        return None, ('Missing authorization token', 401)
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token['uid'], None
    except auth.InvalidIdTokenError:
        return None, ('Invalid ID token', 403)
    except Exception as e:
        print(f"Error verifying token: {e}")
        return None, (f'Token verification failed: {e}', 403)

def get_user_storage_path(uid):
    return f"user_uploads/{uid}/docs/"

def get_user_document_chunks_collection_path(uid, document_hash):
    return db.collection('users').document(uid).collection('processed_documents').document(document_hash).collection('chunks')

def get_user_processed_files_metadata_collection_path(uid):
    return db.collection('users').document(uid).collection('processed_files_metadata')

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot_product / (norm_v1 * norm_v2)

# --- Flask App and Routes ---
app = Flask(__name__)
print("INFO (main.py): Flask app initialized.") # Debug print

@app.route('/upload_pdf', methods=['POST'])
def http_upload_pdf():
    current_embeddings_model = get_embedding_model()
    if not current_embeddings_model:
        return jsonify({"error": "Embedding service not available. Check API key or server logs."}), 503

    uid, error_tuple = get_user_id_from_request(request) # Renamed 'error' to 'error_tuple'
    if error_tuple: return jsonify({"error": error_tuple[0]}), error_tuple[1]

    # ... (rest of the /upload_pdf logic is THE SAME as the last full version)
    # Ensure you use `current_embeddings_model.embed_query(...)`
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "No selected file or not a PDF"}), 400

    original_filename = secure_filename(file.filename)
    file_stream = file.stream
    document_hash = get_file_hash(file_stream)

    processed_files_metadata_col = get_user_processed_files_metadata_collection_path(uid)
    doc_meta_ref = processed_files_metadata_col.document(document_hash)

    if doc_meta_ref.get().exists:
        return jsonify({"message": f"File {original_filename} (hash: {document_hash}) already processed."}), 200

    storage_path_prefix = get_user_storage_path(uid)
    blob_name = f"{document_hash}_{original_filename}"
    blob_path = f"{storage_path_prefix}{blob_name}"
    blob = bucket.blob(blob_path)
    file_stream.seek(0)
    blob.upload_from_file(file_stream, content_type='application/pdf')
    print(f"Uploaded {original_filename} to {blob_path} for user {uid}")

    temp_pdf_path = f"/tmp/{blob_name}"
    file_stream.seek(0)
    with open(temp_pdf_path, 'wb') as f_temp:
        f_temp.write(file_stream.read())

    chunk_count_successfully_embedded = 0
    all_langchain_chunks = []
    try:
        loader = PyPDFLoader(temp_pdf_path)
        raw_documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, 
            separators=["\n\n", "\n", ". ", " ", ""], keep_separator=False
        )
        
        for page_doc in raw_documents:
            page_chunks = text_splitter.create_documents([page_doc.page_content], metadatas=[page_doc.metadata])
            all_langchain_chunks.extend(page_chunks)

        chunks_collection_ref = get_user_document_chunks_collection_path(uid, document_hash)
        batch_ops = db.batch() # Renamed to avoid conflict with 'batch' module
        
        for i, chunk_doc_obj in enumerate(all_langchain_chunks):
            text_content_for_embedding = clean_text(chunk_doc_obj.page_content)
            if not text_content_for_embedding: continue

            try:
                embedding_vector = current_embeddings_model.embed_query(text_content_for_embedding)
            except Exception as e:
                print(f"Error generating embedding for chunk {i} of {original_filename}: {e}")
                continue

            chunk_firestore_doc_ref = chunks_collection_ref.document(f"chunk_{i:04d}")
            batch_ops.set(chunk_firestore_doc_ref, {
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
            
            if (chunk_count_successfully_embedded) % 490 == 0:
                batch_ops.commit()
                batch_ops = db.batch()
        
        if (chunk_count_successfully_embedded % 490) != 0 or \
           (chunk_count_successfully_embedded == 0 and any(c.page_content for c in all_langchain_chunks)):
            if chunk_count_successfully_embedded > 0 or any(c.page_content for c in all_langchain_chunks):
                 batch_ops.commit()
        
        print(f"Stored {chunk_count_successfully_embedded} chunks for {original_filename}")

        status = "processed" if chunk_count_successfully_embedded > 0 else "processing_failed_no_chunks_embedded"
        doc_meta_ref.set({
            "original_filename": original_filename, "storage_path": blob_path,
            "document_hash": document_hash, "total_chunks_processed": len(all_langchain_chunks),
            "chunks_successfully_embedded": chunk_count_successfully_embedded, "status": status,
            "uploaded_at": firestore.SERVER_TIMESTAMP, "last_processed_at": firestore.SERVER_TIMESTAMP
        })
        
    except Exception as e:
        print(f"Error processing PDF {original_filename} for user {uid}: {e}")
        import traceback
        traceback.print_exc()
        doc_meta_ref.set({
            "original_filename": original_filename, "document_hash": document_hash,
            "status": "processing_failed", "error": str(e),
            "uploaded_at": firestore.SERVER_TIMESTAMP, "last_processed_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    if chunk_count_successfully_embedded == 0 and len(all_langchain_chunks) > 0:
        return jsonify({"error": "PDF processed, but no text chunks could be embedded.", "document_hash": document_hash}), 500
    return jsonify({
        "message": f"File {original_filename} processed. Stored {chunk_count_successfully_embedded} embeddable chunks.",
        "document_hash": document_hash
    }), 201


@app.route('/chat', methods=['POST'])
def http_chat():
    current_embeddings_model = get_embedding_model()
    current_llm = get_llm()
    if not current_embeddings_model or not current_llm:
        return jsonify({"error": "AI services not available. Check API key or server logs."}), 503

    uid, error_tuple = get_user_id_from_request(request) # Renamed 'error' to 'error_tuple'
    if error_tuple: return jsonify({"error": error_tuple[0]}), error_tuple[1]
    
    # ... (rest of the /chat logic is THE SAME as the last full version)
    # Ensure you use `current_embeddings_model.embed_query(...)` and `chain = LLMChain(llm=current_llm, ...)`
    data = request.get_json()
    query_text = data.get('query')
    frontend_chat_history = data.get('chat_history', [])

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    try:
        query_embedding = current_embeddings_model.embed_query(query_text)
        np_query_embedding = np.array(query_embedding)

        processed_files_col = get_user_processed_files_metadata_collection_path(uid)
        user_doc_metas_query = processed_files_col.where("status", "==", "processed").stream()
        
        all_user_chunks_for_similarity = []
        for doc_meta_snap in user_doc_metas_query:
            doc_hash = doc_meta_snap.id
            chunks_col_ref = get_user_document_chunks_collection_path(uid, doc_hash)
            chunk_docs_stream = chunks_col_ref.stream() 
            for chunk_doc_snap in chunk_docs_stream:
                chunk_data = chunk_doc_snap.to_dict()
                if 'embedding' in chunk_data and 'text_for_retrieval' in chunk_data:
                    all_user_chunks_for_similarity.append({
                        'id': chunk_doc_snap.id, 'text': chunk_data['text_for_retrieval'],
                        'embedding': np.array(chunk_data['embedding']),
                        'original_filename': chunk_data.get('original_filename', 'N/A'),
                        'chunk_index': chunk_data.get('chunk_index', -1),
                        'source_metadata': chunk_data.get('source_metadata', {})
                    })
        
        if not all_user_chunks_for_similarity:
            return jsonify({"answer": "I don't have any searchable documents for you yet."}), 200

        similarities = []
        for chunk_info in all_user_chunks_for_similarity:
            sim = cosine_similarity(np_query_embedding, chunk_info['embedding'])
            similarities.append((sim, chunk_info))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = 3
        
        retrieved_langchain_docs = []
        added_texts_for_context = set()
        for sim_score, chunk_info in similarities:
            if len(retrieved_langchain_docs) >= top_k: break
            if chunk_info['text'] and chunk_info['text'] not in added_texts_for_context:
                doc_metadata = {
                    "source_filename": chunk_info['original_filename'], "chunk_id": chunk_info['id'],
                    "original_chunk_index": chunk_info['chunk_index'],
                    "page_number": chunk_info.get('source_metadata', {}).get('page', 'N/A'),
                    "similarity_score": float(sim_score) 
                }
                retrieved_langchain_docs.append(Document(page_content=chunk_info['text'], metadata=doc_metadata))
                added_texts_for_context.add(chunk_info['text'])

        if not retrieved_langchain_docs:
             return jsonify({"answer": "Found documents, but no specifically relevant info for your query."}), 200

        context_str = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source_filename', 'Unknown')} (Page {doc.metadata.get('page_number', 'N/A')}, Chunk approx. {doc.metadata.get('original_chunk_index', 'N/A')})\nContent: {doc.page_content}" for doc in retrieved_langchain_docs])
        
        chat_history_for_prompt_list = []
        for entry in frontend_chat_history:
            role = "Human" if entry.get("role") == "user" else "Assistant"
            chat_history_for_prompt_list.append(f"{role}: {entry.get('content')}")
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
        chain = LLMChain(llm=current_llm, prompt=prompt) # Use current_llm
        response = chain.invoke({"chat_history": chat_history_str_for_prompt, "context": context_str, "question": query_text})
        answer = response["text"]
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error in /chat for user {uid}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# --- /reset_user_data (same as before, including delete_collection_recursively) ---
def delete_collection_recursively(coll_ref, batch_size=100):
    docs = coll_ref.limit(batch_size).stream()
    deleted_count = 0
    while True:
        current_batch_deleted = 0
        batch_ops = db.batch()
        for doc_snap in docs:
            # Check if this is a document in 'processed_documents' to delete its 'chunks' subcollection
            path_parts = doc_snap.reference.path.split('/')
            # A path to a document in processed_documents is 'users/{uid}/processed_documents/{doc_hash}' (4 parts if we count from users)
            # Or, more simply, check the parent collection ID
            if doc_snap.reference.parent.id == "processed_documents":
                 chunks_subcollection = doc_snap.reference.collection("chunks")
                 delete_collection_recursively(chunks_subcollection, batch_size)

            batch_ops.delete(doc_snap.reference)
            current_batch_deleted += 1
        
        if current_batch_deleted == 0: break
        batch_ops.commit()
        deleted_count += current_batch_deleted
        print(f"Deleted {current_batch_deleted} docs from {coll_ref.path}. Total: {deleted_count}")
        docs = coll_ref.limit(batch_size).stream()
    return deleted_count

@app.route('/reset_user_data', methods=['POST'])
def http_reset_user_data():
    uid, error_tuple = get_user_id_from_request(request) # Renamed 'error' to 'error_tuple'
    if error_tuple: return jsonify({"error": error_tuple[0]}), error_tuple[1]

    try:
        print(f"Initiating data reset for user {uid}...")
        storage_path_prefix = get_user_storage_path(uid)
        blobs = list(bucket.list_blobs(prefix=storage_path_prefix))
        for blob in blobs:
            blob.delete()
            print(f"Deleted {blob.name} from Storage for user {uid}.")

        user_ref = db.collection('users').document(uid)
        processed_files_meta_col = user_ref.collection('processed_files_metadata')
        print(f"Deleting 'processed_files_metadata' for user {uid}...")
        delete_collection_recursively(processed_files_meta_col)

        processed_documents_col = user_ref.collection('processed_documents')
        print(f"Deleting 'processed_documents' (and their 'chunks') for user {uid}...")
        delete_collection_recursively(processed_documents_col)

        print(f"Data reset completed for user {uid}.")
        return jsonify({"message": "User data has been reset."}), 200
    except Exception as e:
        print(f"Error resetting user data for {uid}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to reset user data: {str(e)}"}), 500

# Expose Flask app for Firebase Functions
api = app
print("INFO (main.py): Reached end of main.py. Flask app 'api' is defined.") # Debug print