from fastapi.middleware.cors import CORSMiddleware
"""
cors_config.py
----------------
This module contains the CORS (Cross-Origin Resource Sharing) configuration
for the FastAPI backend. It ensures that frontend applications (like your
HTML/Tailwind UI or Streamlit app) are allowed to communicate with the backend
securely and without CORS-related browser errors.
"""
def setup_cors(app):
    """
    Function to configure CORS for the FastAPI app.
    Keeps your main.py clean and modular.
    """
    origins = [
        "http://localhost:8501",   # Streamlit frontend
        "http://localhost:5500",   # HTML Live Server frontend
        "http://127.0.0.1:5500",
        "https://your-frontend-domain.com",  # Production domain
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )