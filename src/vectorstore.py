import os
import shutil
from pathlib import path 

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.embeddings import get_embedding_model
from src.loaders import load_documents,split_documents

load_dotenv()

DEFAULT_VECTORSTORE_DIR = path("data/vectorstore")
DEFAULT_DOCUMENTS_DIR = ("data/documents")
INDEX_NAME = "index"

