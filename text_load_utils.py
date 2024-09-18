
import streamlit as st

import re
from io import BytesIO
from typing import Any, Dict, List

from pypdf import PdfReader

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

import tempfile


@st.cache_data
def create_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Create text chunks from a large text block."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


@st.cache_data
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


@st.cache_data
def load_default_pdf():
    data = db.storage.binary.get(key="steve-jobs-commencement-pdf")
    with tempfile.NamedTemporaryFile("wb") as f:
        # The pdf is written to a file on disk
        f.write(data)
        # Now we can load it with a filename
        loader = PyPDFLoader(f.name)
        pages = loader.load_and_split()
        return pages