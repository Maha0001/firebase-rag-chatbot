# Firebase Functions for DocuChat RAG

## Overview

This directory contains the Python backend logic for the DocuChat RAG application, deployed as Firebase Cloud Functions. These functions handle:

*   User authentication token verification.
*   PDF document upload, processing (text extraction, chunking), and embedding generation using Google's Generative AI models.
*   Storage of original PDFs in Firebase Cloud Storage.
*   Storage and retrieval of processed document chunks and metadata from Firestore.
*   RAG-based chat functionality, retrieving relevant document chunks and generating responses via an LLM.
*   User data reset capabilities.

## Environment Variables

The following environment variables are critical for the proper functioning of these Cloud Functions:

*   **`GOOGLE_API_KEY`**: This API key is **required** for accessing Google's Generative AI services (e.g., Gemini models for embeddings and chat responses).
    *   Ensure the key is valid and has the necessary permissions for the AI models used (e.g., "models/text-embedding-004", "gemini-2.5-pro-preview-05-06").
    *   Set this variable in your Firebase Functions environment using the Firebase CLI:
        ```bash
        firebase functions:config:set google_api_key="YOUR_ACTUAL_API_KEY"
        ```
    *   After setting the config, you must redeploy your functions for the changes to take effect.

## Intended Code Structure

The backend code is designed with a modular structure for better organization and maintainability:

*   **`main.py`**:
    *   Initializes the Firebase Admin SDK and global clients (Firestore, Storage).
    *   Initializes the Flask application.
    *   Contains lazy-loading functions (`get_embedding_model`, `get_llm`) for AI model clients, using configuration from `config.py`.
    *   **Intended:** Registers the API routes defined in `routes.py`.
    *   Exports the Flask app instance (`api`) for Firebase.
*   **`routes.py`**:
    *   Defines a Flask Blueprint (`api_bp`) for all API endpoints (e.g., `/upload_pdf`, `/chat`, `/reset_user_data`).
    *   Contains the route handler functions, using the `@require_auth` decorator and helper functions from other modules.
*   **`utils.py`**:
    *   Contains general utility functions for tasks like text cleaning, file hashing, Firestore path generation, cosine similarity calculation, and recursive Firestore collection deletion.
*   **`auth_utils.py`**:
    *   Provides authentication-related utilities, primarily the `require_auth` decorator for protecting routes and the `_get_user_id_from_token` helper for verifying Firebase ID tokens.
*   **`config.py`**:
    *   Stores application-wide constants such as AI model names, Firestore batch sizes, text splitting parameters, and feature flags or limits.

## Deployment

To deploy these functions to your Firebase project, use the Firebase CLI:

```bash
firebase deploy --only functions
```

Ensure you have the necessary Firebase project configuration and have authenticated with the Firebase CLI.

## **Important Note on Current Status**

Due to limitations encountered with the development tooling, the main `functions/main.py` file **could not be updated** to remove its original monolithic code and properly register the new `routes.py` Blueprint.

**As a result:**
*   The backend currently operates based on an older, monolithic version of `functions/main.py`.
*   The modularized logic in `routes.py`, `utils.py`, `auth_utils.py`, and `config.py` (while present in the codebase) **is not actively used by the deployed functions.**
*   Any features or optimizations implemented solely within `routes.py` (such as the chat chunk limiting optimization) are **not currently active** in the application.

This is a known issue pending resolution of the tool limitations affecting `functions/main.py`.
