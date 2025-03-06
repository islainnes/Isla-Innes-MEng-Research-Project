import faiss
import json
from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def load_faiss_index(index_path="faiss_index/papers_index", metadata_path="faiss_index/papers_metadata.json"):
    """Load FAISS index and metadata"""
    try:
        # Initialize sentence transformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load index
        print("\nLoading FAISS index...")
        index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {index.ntotal} vectors")
        
        print("Loading metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata for {len(metadata)} papers")
        
        return index, metadata, encoder
            
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        return None, None, None

def query_similar_papers(query, index, metadata, encoder, k=5):
    """Query FAISS index for similar papers"""
    try:
        print(f"\nQuerying FAISS index for: '{query}'\n")
        
        # Encode query
        query_vector = encoder.encode([query])[0]
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # Search index
        D, I = index.search(query_vector, k)
        
        # Get results
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
            if idx < len(metadata):
                paper = metadata[idx]
                paper['similarity'] = float(1 - dist)
                results.append(paper)
        
        return results
            
    except Exception as e:
        print(f"Error querying similar papers: {e}")
        return []

def get_paper_context(topic, num_papers=3):
    """Get formatted context from relevant papers for a given topic"""
    print("\nInitializing FAISS index...")
    index, metadata_papers, encoder = load_faiss_index()
    
    if not all([index, metadata_papers, encoder]):
        return "", []
        
    relevant_papers = query_similar_papers(topic, index, metadata_papers, encoder)
    
    # Format the context
    context = "Based on these relevant papers:\n"
    for i, paper in enumerate(relevant_papers, 1):
        context += f"{i}. {paper['title']} ({paper.get('year', 'N/A')})\n"
        context += f"   Excerpt: {paper.get('excerpt', '')[:200]}...\n\n"
        
    return context, relevant_papers
