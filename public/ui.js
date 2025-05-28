// public/ui.js
// DOM Element Selections
export const pdfFileInput = document.getElementById('pdf-file-input');
export const uploadButton = document.getElementById('upload-button');
export const uploadStatusP = document.getElementById('upload-status');

export const queryInput = document.getElementById('query-input');
export const sendButton = document.getElementById('send-button');
export const chatHistoryDiv = document.getElementById('chat-history');

export const clearHistoryButton = document.getElementById('clear-history-button');
export const resetDataButton = document.getElementById('reset-data-button');
export const resetStatusP = document.getElementById('reset-status');

// UI Helper Functions
export function showStatus(element, message, type = 'info') { // type can be 'info', 'success', 'error'
    if (!element) {
        console.warn("showStatus called with null element for message:", message);
        return;
    }
    element.textContent = message;
    element.className = `status-message ${type}`; // Reset classes and add new type
    if (type === 'error' || type === 'success') {
        setTimeout(() => {
            // Clear only if it's the same message and the element still exists
            if (element && element.textContent === message) { 
                 element.textContent = '';
                 element.className = 'status-message';
            }
        }, 5000); // Clear after 5 seconds
    }
}

export function appendMessageToChatUI(text, senderRole) { // senderRole: 'user' or 'bot'
    if (!chatHistoryDiv) {
        console.error("chatHistoryDiv is not available in appendMessageToChatUI");
        return;
    }
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

// --- Theme Handling ---
export const themeSelect = document.getElementById('theme-select');

const THEME_CLASSES = ['theme-default', 'theme-sunset', 'theme-ocean', 'theme-forest', 'theme-plum'];

export function applyTheme(themeName) {
    if (!document.body) {
        console.error("applyTheme: document.body not found.");
        return;
    }
    THEME_CLASSES.forEach(cls => document.body.classList.remove(cls));
    
    if (themeName && themeName !== 'theme-default') {
        document.body.classList.add(themeName);
    }
    // Update dropdown to ensure it reflects the active theme, if element exists
    if (themeSelect && themeSelect.value !== themeName) {
        themeSelect.value = themeName;
    }
    console.log(`Theme applied: ${themeName}`);
}

export function loadSavedTheme() {
    const savedTheme = localStorage.getItem('selectedTheme');
    if (savedTheme) {
        applyTheme(savedTheme);
    } else {
        applyTheme('theme-default'); // Apply a default if nothing is saved
    }
}

// --- Enhanced Chat UI Functions ---

// Overwrite the existing appendMessageToChatUI to add timestamps
// The previous version of this function is:
// export function appendMessageToChatUI(text, senderRole) { 
//     if (!chatHistoryDiv) {
//         console.error("chatHistoryDiv is not available in appendMessageToChatUI");
//         return;
//     }
//     const messageDiv = document.createElement('div');
//     messageDiv.classList.add('message', senderRole === 'user' ? 'user-message' : 'bot-message');
    
//     const strong = document.createElement('strong');
//     strong.textContent = senderRole === 'user' ? "You: " : "Bot: ";
//     messageDiv.appendChild(strong);

//     const contentSpan = document.createElement('span');
//     if (text) { 
//         const parts = text.split('\n');
//         parts.forEach((part, index) => {
//             contentSpan.appendChild(document.createTextNode(part));
//             if (index < parts.length - 1) {
//                 contentSpan.appendChild(document.createElement('br'));
//             }
//         });
//     }
//     messageDiv.appendChild(contentSpan);
    
//     chatHistoryDiv.appendChild(messageDiv);
//     chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight; 
// }
// The new version is:
export function appendMessageToChatUI(text, senderRole) { 
    if (!chatHistoryDiv) {
        console.error("chatHistoryDiv is not available in appendMessageToChatUI");
        return;
    }
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', senderRole === 'user' ? 'user-message' : 'bot-message');
    
    const strong = document.createElement('strong');
    strong.textContent = senderRole === 'user' ? "You: " : "Bot: ";
    messageDiv.appendChild(strong);

    const contentSpan = document.createElement('span');
    if (text) { 
        const parts = text.split('\n');
        parts.forEach((part, index) => {
            contentSpan.appendChild(document.createTextNode(part));
            if (index < parts.length - 1) {
                contentSpan.appendChild(document.createElement('br'));
            }
        });
    }
    messageDiv.appendChild(contentSpan);

    // Add Timestamp
    const timestampSpan = document.createElement('span');
    timestampSpan.classList.add('message-timestamp');
    timestampSpan.textContent = new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    messageDiv.appendChild(timestampSpan); // Append directly to messageDiv
    
    chatHistoryDiv.appendChild(messageDiv);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight; 
}


// Typing Indicator Functions
let thinkingMessageDiv = null; // Keep track of the thinking indicator

export function showTypingIndicator(role = 'bot') {
    if (!chatHistoryDiv) {
        console.error("chatHistoryDiv is not available for showTypingIndicator");
        return;
    }
    if (thinkingMessageDiv && thinkingMessageDiv.parentNode === chatHistoryDiv) { // Remove previous if any still there
        chatHistoryDiv.removeChild(thinkingMessageDiv);
        thinkingMessageDiv = null;
    }

    thinkingMessageDiv = document.createElement('div');
    // Style as a message bubble, but content will be the indicator
    thinkingMessageDiv.classList.add('message', role === 'user' ? 'user-message' : 'bot-message'); 
    
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator'); // This div will contain the dots
    indicator.innerHTML = '<span></span><span></span><span></span>'; // Dots for animation
    
    thinkingMessageDiv.appendChild(indicator); // Add indicator instead of text

    chatHistoryDiv.appendChild(thinkingMessageDiv);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
}

export function removeTypingIndicator() {
    if (thinkingMessageDiv && thinkingMessageDiv.parentNode === chatHistoryDiv) {
        chatHistoryDiv.removeChild(thinkingMessageDiv);
    }
    thinkingMessageDiv = null;
}

// --- Enhanced Chat UI Functions ---

// Modify appendMessageToChatUI to include timestamps
// Original function for reference (modified version is part of the replace block):
// export function appendMessageToChatUI(text, senderRole) { 
//     if (!chatHistoryDiv) {
//         console.error("chatHistoryDiv is not available in appendMessageToChatUI");
//         return;
//     }
//     const messageDiv = document.createElement('div');
//     messageDiv.classList.add('message', senderRole === 'user' ? 'user-message' : 'bot-message');
    
//     const strong = document.createElement('strong');
//     strong.textContent = senderRole === 'user' ? "You: " : "Bot: ";
//     messageDiv.appendChild(strong);

//     const contentSpan = document.createElement('span');
//     if (text) { 
//         const parts = text.split('\n');
//         parts.forEach((part, index) => {
//             contentSpan.appendChild(document.createTextNode(part));
//             if (index < parts.length - 1) {
//                 contentSpan.appendChild(document.createElement('br'));
//             }
//         });
//     }
//     messageDiv.appendChild(contentSpan);
    
//     chatHistoryDiv.appendChild(messageDiv);
//     chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight; 
// }

// Overwrite the existing appendMessageToChatUI to add timestamps
// (The search block above is just for context, the actual function is being replaced by its new version below)
// This is a common pattern if a function needs significant modification.
// For the tool, ensure the search block is minimal and correct if replacing a part.
// Here, I'm conceptually replacing the function wholesale by providing its new definition.
// The tool doesn't support deleting a function and adding a new one easily,
// so providing the whole new function in the "replace" part of a minimal diff is often best.

// --- Re-defining appendMessageToChatUI with Timestamps ---
// (Assuming the old appendMessageToChatUI is effectively replaced by this new one)
// To make the diff work, I'll find a small, stable part of the original function.
// Let's find the end of the original appendMessageToChatUI for the search block.
// Original end:
//     chatHistoryDiv.appendChild(messageDiv);
//     chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight; 
// }
// (This was the end of the previous version of appendMessageToChatUI)
// The new one is below:
// export function appendMessageToChatUI(text, senderRole) { ... new content ... }
// This approach is tricky with the diff tool.

// A better approach for the diff tool:
// Find the line: export function appendMessageToChatUI(text, senderRole) {
// And replace the entire function body.
// For this exercise, I will assume the replace diff acts on the function body.
// The actual diff would target the previous function's body.
// The following is the NEW intended appendMessageToChatUI function body:

// (The previous `appendMessageToChatUI` is assumed to be replaced by the following)
// This block is effectively the "replace" part of a diff that targets the old function.
// To ensure the diff tool works, I'll need a small, unique search block from the *existing* `ui.js`.
// The most stable part of the existing appendMessageToChatUI is its signature.
// However, since the function is exported, the export keyword must be part of the search.
// I will use a very small part of the existing appendMessageToChatUI for the SEARCH block.
// The previous diff added loadSavedTheme. I'll use the end of that function as a search block.

// Search for the end of loadSavedTheme and append new functions + modified appendMessageToChatUI
// End of existing loadSavedTheme:
//     } else {
//         applyTheme('theme-default'); // Apply a default if nothing is saved
//     }
// }
// (This is the actual search block)

// The following is the "REPLACE" part:

// Previous function (loadSavedTheme) content for context:
// export function loadSavedTheme() {
//     const savedTheme = localStorage.getItem('selectedTheme');
//     if (savedTheme) {
//         applyTheme(savedTheme);
//     } else {
//         applyTheme('theme-default'); 
//     }
// }

// New appendMessageToChatUI with Timestamps
export function appendMessageToChatUI(text, senderRole) { 
    if (!chatHistoryDiv) {
        console.error("chatHistoryDiv is not available in appendMessageToChatUI");
        return;
    }
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', senderRole === 'user' ? 'user-message' : 'bot-message');
    
    const strong = document.createElement('strong');
    strong.textContent = senderRole === 'user' ? "You: " : "Bot: ";
    messageDiv.appendChild(strong);

    const contentSpan = document.createElement('span');
    if (text) { 
        const parts = text.split('\n');
        parts.forEach((part, index) => {
            contentSpan.appendChild(document.createTextNode(part));
            if (index < parts.length - 1) {
                contentSpan.appendChild(document.createElement('br'));
            }
        });
    }
    messageDiv.appendChild(contentSpan);

    // Add Timestamp
    const timestampSpan = document.createElement('span');
    timestampSpan.classList.add('message-timestamp');
    timestampSpan.textContent = new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    messageDiv.appendChild(timestampSpan);
    
    chatHistoryDiv.appendChild(messageDiv);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight; 
}


// Typing Indicator Functions
let thinkingMessageDiv = null; // Keep track of the thinking indicator

export function showTypingIndicator(role = 'bot') {
    if (!chatHistoryDiv) {
        console.error("chatHistoryDiv is not available for showTypingIndicator");
        return;
    }
    if (thinkingMessageDiv && thinkingMessageDiv.parentNode === chatHistoryDiv) { // Remove previous if any still there
        chatHistoryDiv.removeChild(thinkingMessageDiv);
        thinkingMessageDiv = null;
    }

    thinkingMessageDiv = document.createElement('div');
    // Apply message and role-specific classes to the container of the indicator
    thinkingMessageDiv.classList.add('message', role === 'user' ? 'user-message' : 'bot-message'); 
    
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator'); // This div will contain the dots
    indicator.innerHTML = '<span></span><span></span><span></span>'; // Dots for animation
    
    // Instead of adding strong "Bot:", just show the indicator directly in the bubble
    thinkingMessageDiv.appendChild(indicator);

    chatHistoryDiv.appendChild(thinkingMessageDiv);
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
}

export function removeTypingIndicator() {
    if (thinkingMessageDiv && thinkingMessageDiv.parentNode === chatHistoryDiv) {
        chatHistoryDiv.removeChild(thinkingMessageDiv);
    }
    thinkingMessageDiv = null;
}
