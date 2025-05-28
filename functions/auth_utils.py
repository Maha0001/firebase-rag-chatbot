# functions/auth_utils.py
"""
Authentication utilities for the Firebase Cloud Functions backend.

This module provides a decorator (`require_auth`) for Flask routes to ensure
that incoming requests are authenticated via a Firebase ID token. It also
includes the helper function (`_get_user_id_from_token`) used by the decorator
to verify the token and extract the user ID (UID).
"""
import functools
from typing import Callable, Tuple, Optional, Dict, Any

from flask import request, jsonify, Response as FlaskResponse
from firebase_admin import auth

# Define a type alias for the error response tuple for clarity
ErrorResponse = Tuple[Dict[str, str], int]

def _get_user_id_from_token() -> Tuple[Optional[str], Optional[ErrorResponse]]:
    """
    Verifies a Firebase ID token from the 'Authorization' header and extracts the UID.

    This function expects a 'Bearer' token in the 'Authorization' header.
    It uses the `flask.request` object to access request headers.

    Returns:
        A tuple containing:
            - Optional[str]: The user ID (UID) if the token is valid and verified.
                             None if verification fails or token is missing.
            - Optional[ErrorResponse]: A tuple `(error_dict, status_code)` if verification fails.
                                       None if verification is successful.
                                       `error_dict` is a dictionary suitable for `jsonify`.
    """
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        error_msg = "Missing or malformed authorization token. Expected 'Bearer <token>'."
        return None, ({"error": error_msg}, 401)
    
    id_token = auth_header.split('Bearer ')[-1]
    if not id_token:
        return None, ({"error": "Missing authorization token after 'Bearer ' prefix."}, 401)
    
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token['uid'], None
    except auth.InvalidIdTokenError as e:
        print(f"WARN (auth_utils.py): Invalid ID token: {e}") # Log specific error for server records
        return None, ({"error": "Invalid or expired ID token. Please re-authenticate."}, 403)
    except Exception as e:
        # Log the detailed, unexpected error on the server for debugging
        print(f"ERROR (auth_utils.py): Unexpected error verifying token: {e}") 
        # Return a generic error message to the client
        return None, ({"error": "Token verification failed due to an unexpected error."}, 500)

def require_auth(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    A Flask route decorator that enforces Firebase authentication.

    This decorator wraps a Flask route function. It checks for a valid Firebase
    ID token in the request's 'Authorization' header. If the token is valid,
    it extracts the user ID (UID) and passes it as the first argument to the
    wrapped route function.

    If the token is missing, invalid, or expired, the decorator returns an
    appropriate JSON error response (e.g., 401 or 403 status code) and
    does not call the wrapped route function.

    Args:
        f: The Flask route function to be decorated. This function will receive
           the extracted UID as its first positional argument if authentication succeeds.

    Returns:
        The decorated function, which includes the authentication check.
        If authentication fails, this function will return a Flask Response object
        containing a JSON error message and the appropriate HTTP status code.
    """
    @functools.wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        """
        The wrapper function that performs the authentication check.
        """
        uid, error_response_tuple = _get_user_id_from_token()
        
        if error_response_tuple:
            error_dict, status_code = error_response_tuple
            return jsonify(error_dict), status_code
        
        # If UID is successfully obtained, pass it as the first argument to the wrapped function.
        # The route function must be defined to accept this uid argument.
        # e.g., @require_auth
        #       def my_route(uid: str, ...):
        return f(uid, *args, **kwargs) 
    return decorated_function
