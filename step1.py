#!/usr/bin/env python3
import os
import sys
import glob
import re
import json
import logging
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Hardcoded configuration
MARKDOWN_DIR = "./files_mmd"
FAISS_DIR = "./embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('document_indexing.log')
    ]
)
logger = logging.getLogger(__name__)

def extract_metadata(content: str, file_name: str) -> Dict[str, Any]:
    """Extract metadata such as title, authors, year, and abstract from markdown content."""
    metadata = {
        'title': os.path.basename(file_name).replace('.mmd', '').replace('.pdf', ''),
        'authors': [],
        'year': '',
        'abstract': '',
        'id': file_name
    }
    
    # Try to extract title from content (usually at the beginning)
    title_match = re.search(r'^#\s+(.+?)(?=\n\n|\n#)', content, re.MULTILINE)
    if title_match:
        metadata['title'] = title_match.group(1).strip()
    
    # Try to extract authors (look for patterns like "Author1, Author2, and Author3")
    author_patterns = [
        r'(?:^|\n)by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)?)',
        r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)?)\n(?:University|Institute|Department|School)',
        r'^(?:Authors?|AUTHOR[S]?)[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)?)'
    ]
    
    for pattern in author_patterns:
        author_match = re.search(pattern, content, re.MULTILINE)
        if author_match:
            author_text = author_match.group(1)
            # Split by commas and 'and'
            authors = re.split(r',\s+|\s+and\s+', author_text)
            metadata['authors'] = [author.strip() for author in authors if author.strip()]
            break
    
    # Try to extract year (look for 4-digit numbers that could be years)
    year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', content[:2000])
    if year_matches:
        metadata['year'] = year_matches[0]
    
    # Try to extract abstract
    abstract_patterns = [
        r'(?:^|\n)Abstract[:\s]+(.*?)(?=\n\n|\n#)',
        r'(?:^|\n)ABSTRACT[:\s]+(.*?)(?=\n\n|\n#)',
        r'(?:^|\n)Summary[:\s]+(.*?)(?=\n\n|\n#)',
    ]
    
    for pattern in abstract_patterns:
        abstract_match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            # Clean up the abstract
            abstract_text = re.sub(r'\s+', ' ', abstract_text)
            metadata['abstract'] = abstract_text
            break
    
    return metadata

def chunk_document(content: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split document content into smaller overlapping chunks."""
    # Split by paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # If adding this paragraph would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep some overlap between chunks
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
            
        current_chunk += "\n\n" + paragraph
    
    # Add the last chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def read_markdown_files(markdown_dir: str) -> List[Dict[str, Any]]:
    """Read all markdown files, extract metadata, and chunk content for indexing."""
    markdown_files = glob.glob(os.path.join(markdown_dir, "*.mmd"))
    
    chunked_documents = []
    
    for file_path in markdown_files:
        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Extract metadata
            metadata = extract_metadata(content, file_name)
            
            # Chunk the document content
            chunks = chunk_document(content)
            
            # Create a document entry for each chunk
            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    'file_name': file_name,
                    'content': chunk,
                    'path': file_path,
                    'title': metadata['title'],
                    'authors': metadata['authors'],
                    'year': metadata['year'],
                    'abstract': metadata['abstract'],
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
                
            logger.info(f"Processed {file_name}: {len(chunks)} chunks, authors: {metadata['authors']}")
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            
    return chunked_documents

def create_faiss_database(documents: List[Dict[str, Any]], output_dir: str):
    """Create a FAISS database from the chunked markdown documents."""
    if not documents:
        logger.error("No documents to index.")
        return False
        
    # Create embeddings directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the sentence transformer model
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Create embeddings
    logger.info("Creating embeddings for document chunks...")
    embeddings = []
    metadata = []
    
    for doc in tqdm(documents):
        # Create embedding for the document chunk
        embedding = model.encode(doc['content'])
        embeddings.append(embedding)
        
        # Store rich metadata
        metadata.append({
            'file_name': doc['file_name'],
            'path': doc['path'],
            'title': doc['title'],
            'authors': doc['authors'],
            'year': doc['year'],
            'abstract': doc['abstract'],
            'chunk_id': doc['chunk_id'],
            'total_chunks': doc['total_chunks'],
            'excerpt': doc['content'][:500],  # Store first 500 chars as excerpt
            'id': f"{doc['file_name']}_{doc['chunk_id']}"
        })
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    logger.info("Creating FAISS index...")
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Save the index
    faiss_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, faiss_path)
    logger.info(f"FAISS index saved to {faiss_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.npy")
    np.save(metadata_path, metadata)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Also save metadata as JSON for easier inspection
    json_path = os.path.join(output_dir, "metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata also saved as JSON to {json_path}")
    
    # Save statistics
    stats = {
        "total_documents": len(set(doc['file_name'] for doc in documents)),
        "total_chunks": len(documents),
        "average_chunks_per_document": len(documents) / len(set(doc['file_name'] for doc in documents)) if documents else 0,
        "model_name": MODEL_NAME,
        "embedding_dimension": dimension
    }
    
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Index statistics saved to {stats_path}")
    
    return True

def main():
    # Step 1: Read markdown files, extract metadata, and chunk content
    logger.info(f"Reading Markdown files from {MARKDOWN_DIR}")
    chunked_documents = read_markdown_files(MARKDOWN_DIR)
    logger.info(f"Created {len(chunked_documents)} document chunks from {len(set(doc['file_name'] for doc in chunked_documents))} documents.")
    
    # Step 2: Create FAISS database with rich metadata
    logger.info(f"Creating FAISS database in {FAISS_DIR}")
    success = create_faiss_database(chunked_documents, FAISS_DIR)
    if not success:
        logger.error("FAISS database creation failed.")
        sys.exit(1)
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()
