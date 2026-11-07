import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_embedding_model(model_name=None):

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Add it to the .env file."
        )

    selected_model = model_name or os.getenv(
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL
    )

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=selected_model,
            google_api_key=api_key
        )

    except Exception as e:
        raise RuntimeError(
            "Failed to create the Gemini embedding client."
        ) from e

    return embeddings


def get_embedding_model_name():
    return os.getenv("EMBEDDINGS_MODEL",DEFAULT_EMBEDDING_MODEL)