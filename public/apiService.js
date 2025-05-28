// public/apiService.js
const API_BASE_URL = ''; // No base needed if using rewrites like /upload_pdf directly

export async function uploadPDF(file, token) {
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
    return result; // Contains { message: "...", document_hash: "..." } or error
}

export async function sendMessage(query, chatHistory, token) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ query: query, chat_history: chatHistory })
    });

    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || `Chat API error: ${response.status}`);
    }
    return result; // Contains { answer: "..." } or error
}

export async function resetUserData(token) {
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
    return result; // Contains { message: "..." } or error
}
