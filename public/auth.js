// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyAYrHoyHxyEf9UBa7enrNgEk5Xt-FjsJIY",
    authDomain: "docuchat-rag.firebaseapp.com",
    projectId: "docuchat-rag",
    storageBucket: "docuchat-rag.firebasestorage.app",
    messagingSenderId: "568504076157",
    appId: "1:568504076157:web:d3089f15e974df90bfee14",
    measurementId: "G-QTDN1V2HY7"
  };

// Initialize Firebase
const app = firebase.initializeApp(firebaseConfig); // This is auth.js:13 (or similar)
const auth = firebase.auth();

// FirebaseUI config
const uiConfig = {
    // signInSuccessUrl: window.location.href, // Removed to prevent reload interference
    callbacks: {
        signInSuccessWithAuthResult: function(authResult, redirectUrl) {
            // User successfully signed in.
            // Return true to indicate that we have handled the sign-in success.
            // onAuthStateChanged will handle the UI updates.
            return true;
        }
    },
    signInOptions: [
        firebase.auth.EmailAuthProvider.PROVIDER_ID,
        firebase.auth.GoogleAuthProvider.PROVIDER_ID
    ],
    tosUrl: null, 
    privacyPolicyUrl: null 
};

const ui = new firebaseui.auth.AuthUI(auth);

// DOM Elements
const authContainer = document.getElementById('firebaseui-auth-container');
const userInfoDiv = document.getElementById('user-info');
const userEmailSpan = document.getElementById('user-email');
const logoutButton = document.getElementById('logout-button');
const appContentDiv = document.getElementById('app-content');

let currentIdToken = null; 

// --- Auth State Observer ---
auth.onAuthStateChanged(async (user) => {
    if (user) {
        userEmailSpan.textContent = user.email || user.displayName || "User";
        userInfoDiv.style.display = 'block';
        appContentDiv.style.display = 'block';
        authContainer.style.display = 'none'; 

        try {
            currentIdToken = await user.getIdToken(true); 
            console.log("User signed in. Token acquired.");
            document.dispatchEvent(new CustomEvent('authReady', { detail: { token: currentIdToken } }));
        } catch (error) {
            console.error("Error getting ID token:", error);
            auth.signOut(); 
        }

    } else {
        userInfoDiv.style.display = 'none';
        appContentDiv.style.display = 'none';
        authContainer.style.display = 'block';
        ui.start('#firebaseui-auth-container', uiConfig); 
        currentIdToken = null;
        console.log("User signed out.");
        document.dispatchEvent(new CustomEvent('authSignedOut'));
    }
});

// --- Logout ---
logoutButton.addEventListener('click', () => {
    auth.signOut().catch(error => {
        console.error('Sign out error', error);
        alert('Error signing out. Please try again.');
    });
});

// --- Function to get current ID Token (for app.js) ---
async function getIdToken() {
    const currentUser = auth.currentUser;
    if (currentUser) {
        try {
            currentIdToken = await currentUser.getIdToken(true); 
            return currentIdToken;
        } catch (error) {
            console.error("Error refreshing ID token:", error);
            throw new Error("Failed to refresh authentication token.");
        }
    } else {
        throw new Error('No user currently signed in.');
    }
}

// Export getIdToken for use in other modules
export { getIdToken };