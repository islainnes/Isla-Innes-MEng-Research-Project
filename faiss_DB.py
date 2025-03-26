import os
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def extract_metadata_from_nougat_markdown(content):
    """Extract metadata with more flexible patterns"""
    metadata = {
        "title": "",
        "authors": [],
        "year": "",
        "abstract": "",
    }
    
    # Multiple patterns for title
    title_patterns = [
        r'^#\s+(.+)$',
        r'^Title:\s*(.+)$',
        r'^\s*(.+?)\n\*',  # First line before authors
    ]
    
    for pattern in title_patterns:
        title_match = re.search(pattern, content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
            break
    
    # Multiple patterns for authors
    authors_patterns = [
        r'\*([^*]+)\*',
        r'Authors?:\s*(.+?)(?:\n|$)',
        r'By\s+(.+?)(?:\n|$)',
    ]
    
    for pattern in authors_patterns:
        authors_match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        if authors_match:
            authors = [author.strip() for author in authors_match.group(1).split(',')]
            metadata["authors"] = authors
            break
    
    # Multiple patterns for year
    year_patterns = [
        r'\((\d{4})\)',
        r'Year:\s*(\d{4})',
        r'Published:\s*.*?(\d{4})',
    ]
    
    for pattern in year_patterns:
        year_match = re.search(pattern, content)
        if year_match:
            metadata["year"] = year_match.group(1)
            break
    
    # Multiple patterns for abstract
    abstract_patterns = [
        r'(?:Abstract|ABSTRACT)\s*\n+([^\n#]+)',
        r'Summary:\s*\n+([^\n#]+)',
    ]
    
    for pattern in abstract_patterns:
        abstract_match = re.search(pattern, content, re.IGNORECASE)
        if abstract_match:
            metadata["abstract"] = abstract_match.group(1).strip()
            break
    
    return metadata

def chunk_document(content, metadata, chunk_size=1000, chunk_overlap=100):
    """Split document into larger chunks while preserving metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased from 500
        chunk_overlap=100,  # Increased from 50
        length_function=len,
        separators=["\n## ", "\n\n", "\n", " ", ""]
    )
    
    # Split the content into chunks
    chunks = text_splitter.split_text(content)
    
    # Create documents with metadata for each chunk
    documents = []
    for i, chunk in enumerate(chunks):
        # Add chunk information to metadata
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_id": i,
            "total_chunks": len(chunks),
            "chunk_content": chunk[:100] + "..."  # Preview of chunk content
        })
        
        doc = Document(
            page_content=chunk,
            metadata=chunk_metadata
        )
        documents.append(doc)
    
    return documents

def create_faiss_db(markdown_folder="./files_mmd"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking if folder exists: {os.path.exists(markdown_folder)}")
    print(f"Checking if folder is absolute path: {os.path.isabs(markdown_folder)}")

    if not os.path.exists(markdown_folder):
        raise FileNotFoundError(f"The folder {markdown_folder} does not exist at {os.path.abspath(markdown_folder)}")

    all_documents = []
    for filename in os.listdir(markdown_folder):
        if filename.endswith(".mmd"):
            file_path = os.path.join(markdown_folder, filename)
            print(f"Processing file: {file_path}")
            
            # Read the content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from content
            metadata = extract_metadata_from_nougat_markdown(content)
            metadata["source"] = file_path
            
            # If no title was found in content, use filename
            if not metadata["title"]:
                metadata["title"] = filename.replace('.mmd', '')
            
            # Chunk the document and create Document objects
            chunked_documents = chunk_document(content, metadata)
            all_documents.extend(chunked_documents)
            print(f"Added {len(chunked_documents)} chunks from document: {metadata['title']}")

    if not all_documents:
        raise ValueError(f"No documents were processed from {markdown_folder}")

    print(f"Creating FAISS index with {len(all_documents)} chunks")
    vector_store = FAISS.from_documents(all_documents, embedding_model)
    return vector_store

if __name__ == "__main__":
    try:
        vector_store = create_faiss_db()
        vector_store.save_local("faiss_index")
        print("FAISS index created and saved successfully")
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")