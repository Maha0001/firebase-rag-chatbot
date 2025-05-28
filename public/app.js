// DOM Elements
const pdfFileInput = document.getElementById('pdf-file-input');
const uploadButton = document.getElementById('upload-button');
const uploadStatusP = document.getElementById('upload-status');

const queryInput = document.getElementById('query-input');
const sendButton = document.getElementById('send-button');
const chatHistoryDiv = document.getElementById('chat-history');

const clearHistoryButton = document.getElementById('clear-history-button');
const resetDataButton = document.getElementById('reset-data-button');
const resetStatusP = document.getElementById('reset-status');

let localChatHistory = []; // Stores { role: 'user'/'bot', content: '...' } for display

// --- API Base URL (adjust if your functions are deployed under a different base path by Firebase) ---
// For functions deployed via `firebase.json` rewrites, paths are relative to hosting root.
const API_BASE_URL = ''; // No base needed if using rewrites like /upload_pdf directly

// --- Helper to display status messages ---
function showStatus(element, message, type = 'info') { // type can be 'info', 'success', 'error'
    element.textContent = message;
    element.className = `status-message ${type}`; // Reset classes and add new type
    if (type === 'error' || type === 'success') {
        setTimeout(() => {
            if (element.textContent === message) { // Clear only if it's the same message
                 element.textContent = '';
                 element.className = 'status-message';
            }
        }, 5000); // Clear after 5 seconds
    }
}


// --- File Upload ---
uploadButton.addEventListener('click', async () => {
    const file = pdfFileInput.files[0];
    if (!file) {
        showStatus(uploadStatusP, 'Please select a PDF file.', 'error');
        return;
    }
    if (!file.name.toLowerCase().endsWith('.pdf')) { // Case-insensitive check
        showStatus(uploadStatusP, 'Invalid file type. Please select a PDF.', 'error');
        return;
    }

    showStatus(uploadStatusP, 'Uploading and processing... This may take a moment.', 'info');
    uploadButton.disabled = true;
    pdfFileInput.disabled = true;

    try {
        const token = await getIdToken(); // From auth.js
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/upload_pdf`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
                // 'Content-Type' is set automatically by FormData for multipart
            },
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || `Upload failed with status: ${response.status}`);
        }
        showStatus(uploadStatusP, result.message || 'PDF processed successfully!', 'success');
        pdfFileInput.value = ''; // Clear file input

    } catch (error) {
        console.error('Upload error:', error);
        showStatus(uploadStatusP, `Upload failed: ${error.message}`, 'error');
    } finally {
        uploadButton.disabled = false;
        pdfFileInput.disabled = false;
    }
});

// --- Chat ---
function appendMessageToChatUI(text, senderRole) { // senderRole: 'user' or 'bot'
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', senderRole === 'user' ? 'user-message' : 'bot-message');
    
    const strong = document.createElement('strong');
    strong.textContent = senderRole === 'user' ? "You: " : "Bot: ";
    messageDiv.appendChild(strong);

    // Securely append text content, handling newlines by creating <br> elements
    const contentSpan = document.createElement('span');
    if (text) { // Ensure text is not null or undefined
        const parts = text.split('\n');
        parts.forEach((part, index) => {
            contentSpan.appendChild(document.createTextNode(part));
            if (index < parts.length - 1) {
                contentSpan.appendChild(document.createElement('br'));
            }
        });
    }
    messageDiv.appendChild(contentSpan);
    
    chatHistoryDiv.appendChild(messageDiv);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight; // Auto-scroll to bottom
}

sendButton.addEventListener('click', handleSendMessage);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { // Send on Enter, allow Shift+Enter for new line
        e.preventDefault();
        handleSendMessage();
    }
});

async function handleSendMessage() {
    const query = queryInput.value.trim();
    if (!query) return;

    appendMessageToChatUI(query, 'user');
    localChatHistory.push({ role: 'user', content: query }); // Add to local history for sending to backend
    queryInput.value = ''; // Clear input
    sendButton.disabled = true;
    queryInput.disabled = true;
    appendMessageToChatUI("Thinking...", 'bot'); // Show thinking indicator

    try {
        const token = await getIdToken(); // From auth.js

        // Send the current query and the relevant part of chat history (that LLM expects)
        // The backend expects chat_history as list of {"role": "user/assistant", "content": "..."}
        // We are sending the history that led to this query.
        const backendChatHistory = localChatHistory.slice(0, -1); // Exclude current user query

        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ query: query, chat_history: backendChatHistory })
        });
        
        // Remove "Thinking..." message
        if (chatHistoryDiv.lastChild && chatHistoryDiv.lastChild.textContent.includes("Thinking...")) {
            chatHistoryDiv.removeChild(chatHistoryDiv.lastChild);
        }

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || `Chat API error: ${response.status}`);
        }

        const botResponse = result.answer || "Sorry, I couldn't get a response.";
        appendMessageToChatUI(botResponse, 'bot');
        localChatHistory.push({ role: 'bot', content: botResponse }); // Add bot response to local history

    } catch (error) {
        console.error('Chat error:', error);
        if (chatHistoryDiv.lastChild && chatHistoryDiv.lastChild.textContent.includes("Thinking...")) {
             chatHistoryDiv.removeChild(chatHistoryDiv.lastChild);
        }
        appendMessageToChatUI(`Error: ${error.message}`, 'bot');
        // Optionally remove the last user message from localChatHistory if server call failed and no bot response added
        // localChatHistory.pop(); // If you want to allow user to retry same message
    } finally {
        sendButton.disabled = false;
        queryInput.disabled = false;
        queryInput.focus();
    }
}

// --- Controls ---
clearHistoryButton.addEventListener('click', () => {
    localChatHistory = [];
    chatHistoryDiv.innerHTML = ''; // Clear displayed chat
    appendMessageToChatUI("Chat display cleared.", 'bot');
});

resetDataButton.addEventListener('click', async () => {
    if (!confirm("Are you sure you want to reset all your PDF data on the server? This cannot be undone.")) {
        return;
    }

    showStatus(resetStatusP, 'Resetting data...', 'info');
    resetDataButton.disabled = true;
    try {
        const token = await getIdToken(); // From auth.js
        const response = await fetch(`${API_BASE_URL}/reset_user_data`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || `Reset API error: ${response.status}`);
        }

        showStatus(resetStatusP, result.message || 'Data reset successfully.', 'success');
        localChatHistory = []; // Also clear frontend chat display
        chatHistoryDiv.innerHTML = '';
        appendMessageToChatUI("Your document data has been reset on the server.", 'bot');
        uploadStatusP.textContent = ''; // Clear any old upload status

    } catch (error) {
        console.error('Reset data error:', error);
        showStatus(resetStatusP, `Error resetting data: ${error.message}`, 'error');
    } finally {
        resetDataButton.disabled = false;
    }
});

// Initial state message (can be shown if user is logged in but app content is hidden initially)
// Or moved to auth.js when app-content is shown
document.addEventListener('authReady', () => {
    if (chatHistoryDiv.children.length === 0) { // Only if chat is empty
        appendMessageToChatUI("Welcome! Please upload PDFs and then ask questions about their content.", 'bot');
    }
});
document.addEventListener('authSignedOut', () => {
    chatHistoryDiv.innerHTML = ''; // Clear chat when user signs out
    localChatHistory = [];
    uploadStatusP.textContent = '';
    resetStatusP.textContent = '';
});

// --- Theme Handling Initialization ---
// Imports from ui.js (conceptual, as app.js is the one importing)
// import { themeSelect, applyTheme, loadSavedTheme } from './ui.js'; // This is how app.js would do it

// Load saved theme on startup
// This assumes loadSavedTheme and themeSelect are available (exported from ui.js and imported here)
if (typeof loadSavedTheme === 'function') {
    loadSavedTheme();
} else {
    console.warn("loadSavedTheme function not found. Ensure it's exported from ui.js and imported correctly in app.js if app.js is modular.");
}

// Event listener for theme selection
// This assumes themeSelect is available (exported from ui.js and imported here)
if (typeof themeSelect !== 'undefined' && themeSelect) { 
    themeSelect.addEventListener('change', (event) => {
        if (typeof applyTheme === 'function') {
            const selectedTheme = event.target.value;
            applyTheme(selectedTheme);
            localStorage.setItem('selectedTheme', selectedTheme);
        } else {
            console.warn("applyTheme function not found during theme change event.");
        }
    });
} else {
    // This might be normal if app.js is not yet refactored to import themeSelect
    // or if the element genuinely isn't in the DOM (which would be an HTML/ui.js issue).
    console.warn("themeSelect element not found or not imported for event listener setup.");
}