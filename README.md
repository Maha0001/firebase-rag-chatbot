# DocuChat RAG - Firebase Edition

## Overview

DocuChat RAG is a web application that allows users to upload PDF documents, process them into searchable chunks with embeddings, and interact with a RAG (Retrieval Augmented Generation) powered chat interface to ask questions about the document content. This version is built using Firebase for backend services (Cloud Functions, Firestore, Storage) and a vanilla JavaScript frontend.

## Features Implemented & Status

### 1. Core Functionality
*   **User Authentication:**
    *   Email/Password Sign-in: **Working.** (Fix applied for existing users being pushed to signup).
    *   Google Sign-In: **Working, pending user configuration.** (Guidance provided to configure OAuth credentials in Google Cloud Console).
    *   User data is isolated per authenticated user.
*   **PDF Upload & Processing:**
    *   Users can upload PDF files.
    *   PDFs are processed, text is chunked, and embeddings are generated using Google's Generative AI.
    *   Chunks and metadata are stored in Firestore, original PDFs in Cloud Storage.
    *   **Status:** **Working, pending user API key configuration.** (Fix applied for `firebase.json` rewrites; requires `GOOGLE_API_KEY` to be correctly set for Firebase Functions for embedding generation).
*   **Chat Interface:**
    *   Users can ask questions about their uploaded documents.
    *   The system retrieves relevant chunks and uses a Google Generative AI model to generate answers.
    *   **Status:** Core chat logic is implemented. **Performance optimization (chunk limiting) is implemented in `functions/routes.py` but NOT CURRENTLY ACTIVE due to issues updating `functions/main.py`.**

### 2. Bug Fixes
*   **Email Sign-In Loop:** Resolved an issue where existing users attempting email sign-in were incorrectly prompted for new signup details. This was fixed by modifying `public/auth.js` to use the `callbacks.signInSuccessWithAuthResult` in the FirebaseUI config, preventing a disruptive page reload. (Status: **Working**)
*   **Google Sign-In Issues:** Investigated and provided detailed guidance for users to correctly configure OAuth 2.0 Client ID settings (Authorized JavaScript Origins and Redirect URIs) in their Google Cloud Console. This is a common cause for Google Sign-In failures. (Status: **Working, pending user configuration**)
*   **PDF Upload Failures (Routing):** Corrected `firebase.json` hosting rewrite rules to ensure API calls (`/upload_pdf`, `/chat`, `/reset_user_data`) are correctly routed to the backend Firebase Function. Also emphasized the need for a valid `GOOGLE_API_KEY` for the embedding service. (Status: **Working, pending user API key check**)

### 3. Code Refactoring Efforts
*   **Python Backend (Firebase Functions):**
    *   **Intended Structure:** The backend code was planned to be modularized into:
        *   `config.py`: For application constants (model names, batch sizes).
        *   `auth_utils.py`: For authentication helpers (e.g., `require_auth` decorator).
        *   `utils.py`: For general utility functions (text processing, hashing, Firestore path generation).
        *   `routes.py`: For Flask Blueprint defining API route handlers.
        *   `main.py`: For Firebase/Flask app initialization, global client setup, and Blueprint registration.
    *   **Current Status:** **PARTIALLY IMPLEMENTED / INACTIVE.** Files `config.py`, `auth_utils.py`, `utils.py`, and `routes.py` were successfully created with the refactored logic. However, due to persistent tool limitations preventing updates to `functions/main.py`, this file **could not be refactored** to remove the old monolithic code and register the new `routes.py` Blueprint. **Therefore, the new modular backend structure and any logic within `functions/routes.py` (like the chat optimization) are NOT CURRENTLY ACTIVE.** The application currently runs on the older, monolithic version of `functions/main.py`.
*   **JavaScript Frontend:**
    *   **XSS Fix:** The `appendMessageToChatUI` function (now in `public/ui.js`) was refactored to securely handle text content and newlines by creating TextNodes and `<br>` elements, mitigating potential XSS risks from bot responses. (Status: **Working**)
    *   **Intended Structure:** The frontend was planned to be modularized into:
        *   `ui.js`: For DOM element selections and UI manipulation functions.
        *   `apiService.js`: For handling API fetch calls.
        *   `app.js`: As the main controller orchestrating UI and API interactions.
        *   `auth.js`: (Already existing) for authentication logic.
    *   **Current Status:** **PARTIALLY IMPLEMENTED / INACTIVE.** Files `ui.js` and `apiService.js` were successfully created. `auth.js` was updated to export `getIdToken`, and `index.html` was updated to use ES module script tags. However, due to persistent tool limitations preventing updates to `public/app.js`, this file **could not be refactored** to import from and utilize the new `ui.js` and `apiService.js` modules. **Therefore, the frontend modularization is NOT CURRENTLY ACTIVE.** The application currently runs on the older, monolithic version of `public/app.js`.

### 4. New Features & Enhancements
*   **Chat Optimization (Backend - Chunk Limiting):**
    *   **Description:** Implemented a strategy in `functions/routes.py` to limit the total number of chunks retrieved and processed for similarity scoring in the chat function. This involves a configurable `MAX_CHUNKS_FOR_SIMILARITY_SCORING` and an attempt to prioritize recently processed documents.
    *   **Status:** **Implemented but NOT ACTIVE.** This logic resides in `functions/routes.py`, which is not currently used by `functions/main.py` (see Python Backend Refactoring status).
*   **Gradient Color Themes:**
    *   **Description:** Introduced a theme selection feature allowing users to choose from several gradient color schemes (Sunset, Ocean Breeze, Forest Walk, Plum Velvet, Default). Themes affect body background, headers, and buttons. User preference is saved in `localStorage`.
    *   **Status:** **PARTIALLY ACTIVE / POTENTIALLY BROKEN.**
        *   CSS (`public/style.css`): All necessary CSS variables and theme classes were successfully added. The visual styles for themes *should* be available if the CSS file was correctly updated by the tool in the final attempt.
        *   HTML (`public/index.html`): The theme selector dropdown was successfully added.
        *   JS (`public/ui.js`): Functions `applyTheme` and `loadSavedTheme` were successfully implemented.
        *   JS (`public/app.js`): Logic to handle theme selection changes and load saved themes was added. However, due to the inability to refactor `app.js` to properly import from `ui.js` (see JavaScript Frontend Refactoring status), the interactivity of the theme selector (changing themes, saving preferences) is **likely non-functional.**

*   **Chat Room UI Perfection:**
    *   **Description:** Planned enhancements for message bubble styling, timestamps, a typing indicator, and code snippet formatting.
    *   **Status:** **PARTIALLY IMPLEMENTED / MOSTLY INACTIVE.**
        *   **Timestamps & Typing Indicator Logic (`public/ui.js`):** The JavaScript functions `appendMessageToChatUI` (modified for timestamps) and `showTypingIndicator`/`removeTypingIndicator` were successfully implemented in `public/ui.js`.
        *   **CSS (`public/style.css`):** Could **not** apply the new CSS for enhanced message bubbles, timestamps, the typing indicator, or code snippets due to tool limitations with `public/style.css`.
        *   **JS (`public/app.js`):** Could **not** update `handleSendMessage` to use the new typing indicator functions or add textarea auto-resize logic due to tool limitations with `public/app.js`.
        *   **Result:** While core JS logic for timestamps and typing indicators exists in `ui.js`, the necessary styling is missing, and the integration into `app.js` is incomplete, rendering these features largely inactive or visually unpolished.

## Setup & Running Instructions (High-Level)

1.  **Firebase Project:** Set up a Firebase project with Firestore, Firebase Storage, and Firebase Authentication (Email/Password and Google providers enabled).
2.  **Environment Variables:**
    *   Crucially, set the `GOOGLE_API_KEY` for the Firebase Functions environment. This key must have access to Google's Generative AI services (e.g., for Gemini models used in embeddings and chat). Set this via `firebase functions:config:set google_api_key="YOUR_API_KEY"` and redeploy functions.
    *   Other configurations (Firebase project ID, etc.) are usually handled by the Firebase SDKs if deployed within the Firebase environment.
3.  **Deployment:**
    *   Deploy hosting and functions using the Firebase CLI: `firebase deploy`
    *   To deploy only functions: `firebase deploy --only functions`
    *   To deploy only hosting: `firebase deploy --only hosting`

## Known Issues & Limitations

*   **Backend Refactoring Incomplete:** The primary Firebase Function logic in `functions/main.py` **could not be updated** to use the new modular structure (`routes.py`, `utils.py`, etc.) due to persistent failures with the development tool's file writing capabilities for this specific file. As a result, the backend currently operates on an older, monolithic version of `main.py`. **This means that backend features implemented in `routes.py` (such as the chat optimization) are NOT ACTIVE.**
*   **Frontend Modularization Incomplete:** The main frontend JavaScript file `public/app.js` **could not be updated** to import from and utilize the new `ui.js` and `apiService.js` modules due to tool limitations. **This means the frontend modularization is NOT ACTIVE**, and `app.js` operates in its older, monolithic form.
*   **Theme Interactivity Likely Broken:** While CSS for themes and theme-switching logic in `ui.js` were implemented, the necessary updates to `app.js` to handle theme selection events and make the feature interactive were likely ineffective due to the inability to properly refactor `app.js`.
*   **Chat UI Enhancements Incomplete:**
    *   CSS for improved message bubbles, timestamps, typing indicator, and code snippets **could not be applied** to `public/style.css` due to tool limitations.
    *   JavaScript in `app.js` for integrating the typing indicator and textarea auto-resize **could not be applied**.
    *   While `ui.js` contains some improved logic (timestamps, typing indicator functions), these are not fully styled or integrated.

These limitations are primarily due to issues encountered with the automated development tooling's ability to modify certain key files (`functions/main.py`, `public/app.js`, `public/style.css`).
