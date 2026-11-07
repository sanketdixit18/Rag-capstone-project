from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS={".pdf",".docx"}


def load_documents(folder_path:str)->list[Document]:
    folder=Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found :{folder_path}")


    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise ValueError("No supported document found. Please upload the file in pdf or docx format.")

    documents:list[Document]=[]
    for file_path in files:
        loaded=_load_single_document(file_path)
        documents.extend(loaded)

    documents=[doc for doc in documents if doc.page_content and doc.page_content.strip()]

    if not documents:
        raise ValueError("Document were found but they appear unreadble or empty")

def _load_single_document(file_path : str) -> list[Document]:
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".pdf":
            loader=PyPDFLoader(str(file_path))
            docs=loader.load()

        elif suffix == ".docx":
            loader=Docx2txtLoader(str(file_path))
            docs=loader.load()

        else:
            raise ValueError(f"Unsupported file type : {file_path:name}")

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Count not read {file_path.name}") from e

    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["filename"] = file_path.name

        if "page" in doc.metadata and doc.metadata["page"] is not None:
            try:
                doc.metadata["page"]=int(doc.metadata["page"]) + 1
            except (TypeError,ValueError):
                doc.metadata["page"] = None
            else:
                doc.metadata["page"] = None

        return docs

# Function for the split document 

def split_documents(documents: list[Document],
                    chunk_size: int = 1000,
                    chunk_overlap : int = 150):
    if not documents:
        raise ValueError("No document to split")

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",""]
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produce no chunks.")

    return chunks 

def count_source_files(folder_path: str) -> int:
    """
    Count supported PDF/DOCX files currently stored in the documents folder.

    Args:
        folder_path (str): Path to the document folder.

    Returns:
        int: Number of supported files found.

    Example:
        print(count_source_files("data/documents"))
    """
    folder = Path(folder_path)
    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
