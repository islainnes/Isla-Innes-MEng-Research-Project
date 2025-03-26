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

def consolidate_chunks(chunks):
    """Consolidate chunks from the same paper"""
    papers = {}
    for chunk in chunks:
        paper_id = f"{chunk['title']}_{chunk['source']}"
        if paper_id not in papers:
            papers[paper_id] = {
                'title': chunk['title'],
                'authors': chunk['authors'],
                'year': chunk['year'],
                'abstract': chunk['abstract'],
                'source': chunk['source'],
                'chunks': [],
                'similarity': chunk['similarity']  # Use highest chunk similarity
            }
        papers[paper_id]['chunks'].append({
            'content': chunk['page_content'],
            'chunk_id': chunk['chunk_id'],
            'similarity': chunk['similarity']
        })
    
    # Sort chunks by chunk_id and combine content
    results = []
    for paper in papers.values():
        sorted_chunks = sorted(paper['chunks'], key=lambda x: x['chunk_id'])
        paper['content'] = '\n'.join(chunk['content'] for chunk in sorted_chunks)
        paper['chunks'] = sorted_chunks  # Keep chunk information for reference
        results.append(paper)
    
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

def query_similar_papers(query, index, metadata, encoder, k=15):  # Increased k to get more chunks
    """Query FAISS index for similar chunks"""
    try:
        print(f"\nQuerying FAISS index for: '{query}'\n")
        
        query_vector = encoder.encode([query])[0]
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        D, I = index.search(query_vector, k)
        
        # Get chunks
        chunks = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(metadata):
                chunk = metadata[idx].copy()
                chunk['similarity'] = float(1 - dist)
                chunks.append(chunk)
        
        # Consolidate chunks into papers
        results = consolidate_chunks(chunks)
        
        return results[:5]  # Return top 5 papers after consolidation
            
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
    
    # Format the context with most relevant chunks
    context = "Based on these relevant papers:\n"
    for i, paper in enumerate(relevant_papers[:num_papers], 1):
        context += f"{i}. {paper['title']} ({paper.get('year', 'N/A')})\n"
        # Include the most relevant chunks
        top_chunks = sorted(paper['chunks'], key=lambda x: x['similarity'])[:2]
        for chunk in top_chunks:
            context += f"   Relevant excerpt: {chunk['content'][:200]}...\n"
        context += "\n"
        
    return context, relevant_papers[:num_papers]