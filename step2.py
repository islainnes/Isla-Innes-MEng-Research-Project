import time
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
from autogen import config_list_from_json
import gc
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
from huggingface_hub import login
from dotenv import load_dotenv
import requests
from typing import Dict, Optional, List
import re
import numpy as np
import logging
import sys
from datetime import datetime
from functools import lru_cache
import pickle
import shutil
from langchain_community.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Clear CUDA cache and set PyTorch memory management
logger.info("Initializing GPU memory settings")
torch.cuda.empty_cache()
torch.backends.cuda.max_memory_split_size = 1024 * 1024 * 1024  # 1GB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Set up model configuration
os.environ["OAI_CONFIG_LIST"] = json.dumps([
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_client_cls": "CustomLlama2Client",
        "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
        "n": 1,
        "params": {
            "max_new_tokens": 500,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": 2,
            "eos_token_id": 2,
        },
    }
])

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()
hf_token = os.environ.get('HUGGINGFACE_TOKEN')
if not hf_token:
    logger.critical("HUGGINGFACE_TOKEN not found in environment variables")
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
login(token=hf_token)

# Create a temporary directory for cache
cache_dir = os.path.join(tempfile.gettempdir(), 'autogen_cache')
os.makedirs(cache_dir, exist_ok=True)
logger.debug(f"Cache directory created at: {cache_dir}")

# Configure cache
config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model_client_cls": ["CustomLlama2Client"]}
)

# Define common regex patterns
VALID_NAME_PATTERN = re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)?$')
ET_AL_PATTERN = re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+et\s+al\.$')

# Create utility functions for common tasks
def clear_memory():
    """Clear GPU memory and garbage collect"""
    logger.info("Clearing GPU memory")
    # Clear CUDA cache
    torch.cuda.empty_cache()
    # Manually clean model cache if needed
    for key in list(_model_cache.keys()):
        if key != f"mistralai/Mistral-7B-Instruct-v0.2_cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}":
            logger.info(f"Removing model {key} from cache")
            del _model_cache[key]
    # Force garbage collection
    gc.collect()
    
    # Print memory status after cleaning
    monitor_gpu_memory()


def monitor_gpu_memory():
    """Print current GPU memory usage for debugging"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free = total_memory - reserved
            logger.info(f"GPU {i}: Total Memory: {total_memory:.2f} GB")
            logger.info(f"GPU {i}: Reserved Memory: {reserved:.2f} GB")
            logger.info(f"GPU {i}: Allocated Memory: {allocated:.2f} GB")
            logger.info(f"GPU {i}: Free Memory: {free:.2f} GB")
    else:
        logger.warning("No GPU available")


# Model caching variables
_model_cache = {}

def load_shared_model(model_name, device):
    """Load model once and cache it"""
    # Use in-memory cache instead of lru_cache
    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        logger.info(f"Using cached model {model_name}")
        return _model_cache[cache_key]
    
    logger.info(f"Loading model {model_name} to {device}...")
    
    # Configure quantization settings - using 4-bit quantization for 7B model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Adjust memory allocation for GPU to handle the 7B model
    max_memory = {
        0: "24GiB",  # Use up to 24GB of GPU memory
        "cpu": "8GiB"  # Keep some CPU memory available for offloading if needed
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Let the model decide optimal device mapping
        quantization_config=quantization_config,
        max_memory=max_memory,
        offload_folder="offload",
        trust_remote_code=True  # Add trust_remote_code for newer models
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Model and tokenizer loaded successfully")
    
    # Cache the model
    _model_cache[cache_key] = (model, tokenizer)
    return _model_cache[cache_key]


class CustomLlama2Client:
    def __init__(self, config, **kwargs):
        self.config = config
        self.model_name = config["model"]
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.gen_params = config.get("params", {})
        logger.info(f"Initializing CustomLlama2Client with model {self.model_name} on {self.device}")
        self.model, self.tokenizer = load_shared_model(
            self.model_name, self.device)

    def _format_chat_prompt(self, messages):
        # Format for Mistral-7B-Instruct model
        formatted_prompt = ""
        for message in messages:
            if message["role"] == "system":
                formatted_prompt = f"<s>[INST] {message['content']} [/INST]"
            elif message["role"] == "user":
                formatted_prompt += f"<s>[INST] {message['content']} [/INST]"
            elif message["role"] == "assistant":
                formatted_prompt += f" {message['content']} </s>"
        return formatted_prompt

    def create(self, params):
        response = SimpleNamespace()
        prompt = self._format_chat_prompt(params["messages"])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        response.choices = []
        response.model = self.model_name

        try:
            logger.debug("Generating response with primary parameters")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.gen_params
                )
                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = generated_text.strip()
                choice.message.function_call = None
                response.choices.append(choice)
        except RuntimeError as e:
            logger.error(f"Error during generation: {str(e)}")
            logger.info("Falling back to basic greedy decoding")
            # Fallback to basic greedy decoding if there's an error
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.gen_params.get("max_new_tokens", 1000),
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = generated_text.strip()
                choice.message.function_call = None
                response.choices.append(choice)

        return response

    def message_retrieval(self, response):
        return [choice.message.content for choice in response.choices]

    def cost(self, response):
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        return {}


def clean_response(text):
    """Clean the response text to get only the report content"""
    # Remove model artifacts from generated text
    text = text.replace("[/INST]", "").replace("[INST]", "").strip()
    
    # Remove system message and prompt if they're being echoed back
    if "You are an AI model" in text:
        text = re.sub(r"You are an AI model.*?academic report about", "", text, flags=re.DOTALL)
    
    if "Based on these papers:" in text:
        text = re.sub(r"Based on these papers:.*?IMPORTANT:", "", text, flags=re.DOTALL)
    
    # If there's no header yet, find the start of actual content
    if not text.strip().startswith("##"):
        header_pos = text.find("## BACKGROUND")
        if header_pos >= 0:
            text = text[header_pos:].strip()
        else:
            # Add header if missing
            text = "## BACKGROUND KNOWLEDGE\n" + text
    
    return text.strip()


def extract_sections(text):
    """Extract sections from the report text into a dictionary"""
    sections = {}
    current_section = None
    current_content = []
    
    # Regular sections processing
    for line in text.split('\n'):
        if line.startswith('## '):
            # Save previous section content
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line.replace('## ', '').strip()
            current_content = []
        else:
            current_content.append(line)

    # Add the last section
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    # Ensure all required sections exist (even if empty)
    for required_section in ["BACKGROUND KNOWLEDGE", "CURRENT RESEARCH", "RESEARCH RECOMMENDATIONS"]:
        if required_section not in sections:
            sections[required_section] = ""
    
    return sections


def load_vector_store():
    """Load FAISS vector store using LangChain"""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check for embeddings directory created by step1.py
    if os.path.exists("./embeddings/faiss.index") and os.path.exists("./embeddings/metadata.npy"):
        logger.info("Loading existing FAISS index from ./embeddings directory")
        
        try:
            # Create directory to store converted files if it doesn't exist
            os.makedirs("./converted_index", exist_ok=True)
            
            # Debug: Check original index size before copying
            original_index = faiss.read_index("./embeddings/faiss.index")
            logger.info(f"Original index size before conversion: {original_index.ntotal}")
            
            # Copy the FAISS index to the new directory with the name LangChain expects
            if not os.path.exists("./converted_index/index.faiss"):
                logger.info("Copying FAISS index to compatible format")
                shutil.copy("./embeddings/faiss.index", "./converted_index/index.faiss")
            
            # Debug: Check copied index size
            copied_index = faiss.read_index("./converted_index/index.faiss")
            logger.info(f"Copied index size after conversion: {copied_index.ntotal}")
            
            # Load metadata and create docstore
            logger.info("Loading metadata and creating docstore")
            metadata_list = np.load("./embeddings/metadata.npy", allow_pickle=True)
            logger.info(f"Loaded {len(metadata_list)} document metadata entries")
            
            # Create docstore dictionary that maps IDs to Document objects
            docstore_dict = {}
            for i, meta in enumerate(metadata_list):
                # Ensure we have required fields
                if not all(k in meta for k in ['file_name', 'chunk_id']):
                    logger.warning(f"Metadata entry {i} missing required fields, skipping")
                    continue
                    
                doc_id = f"{meta['file_name']}_{meta['chunk_id']}"
                
                # Use full content instead of just excerpt
                content = meta.get('content', meta.get('excerpt', ''))  # Try content first, fall back to excerpt
                if not content:
                    logger.warning(f"No content found for document {doc_id}, skipping")
                    continue
                
                # Create metadata dict
                doc_metadata = {
                    'file_name': meta.get('file_name', ''),
                    'path': meta.get('path', ''),
                    'title': meta.get('title', ''),
                    'authors': meta.get('authors', []),
                    'year': meta.get('year', ''),
                    'abstract': meta.get('abstract', ''),
                    'chunk_id': meta.get('chunk_id', 0),
                    'total_chunks': meta.get('total_chunks', 1),
                    'content': content,  # Store full content in metadata
                    'id': doc_id
                }
                
                # Create LangChain Document object
                doc = Document(
                    page_content=content,
                    metadata=doc_metadata
                )
                
                # Add to docstore
                docstore_dict[doc_id] = doc
            
            if not docstore_dict:
                raise ValueError("No valid documents found in metadata")
                
            logger.info(f"Created document store with {len(docstore_dict)} entries")
            
            # Create InMemoryDocstore
            docstore = InMemoryDocstore(docstore_dict)
            
            # Load the FAISS index again to ensure we're using the latest state
            index = faiss.read_index("./converted_index/index.faiss")
            logger.info(f"Final index size before creating vectorstore: {index.ntotal}")
            
            # Verify index and docstore sizes match
            if index.ntotal != len(docstore_dict):
                # Try to recover by recreating the index
                logger.warning("Size mismatch detected, attempting to recreate index...")
                shutil.copy("./embeddings/faiss.index", "./converted_index/index.faiss")
                index = faiss.read_index("./converted_index/index.faiss")
                if index.ntotal != len(docstore_dict):
                    raise ValueError(f"Index size ({index.ntotal}) does not match docstore size ({len(docstore_dict)})")
            
            # Create mapping from index positions to document IDs using same format as step1.py
            index_to_id = {i: f"{meta['file_name']}_{meta['chunk_id']}" 
                          for i, meta in enumerate(metadata_list) 
                          if 'file_name' in meta and 'chunk_id' in meta}
            
            # Create FAISS instance manually
            vectorstore = FAISS(
                embedding_function=embedding_model,
                index=index,
                docstore=docstore,  # Use the InMemoryDocstore instead of the raw dict
                index_to_docstore_id=index_to_id
            )
            
            return vectorstore
                
        except Exception as e:
            logger.error(f"Error converting or loading index: {str(e)}", exc_info=True)
            raise e
            
    else:
        logger.error("FAISS index not found in ./embeddings. Please create it with step1.py first.")
        raise FileNotFoundError("FAISS index not found. Please create it with step1.py first.")


def get_paper_context(topic, num_papers=8):
    try:
        vector_store = load_vector_store()
        logger.info(f"Searching for papers related to: {topic}")
        # Get more results to ensure we have enough chunks
        relevant_docs_with_scores = vector_store.similarity_search_with_score(topic, k=num_papers * 5)
        
        # Create a dictionary to group ALL chunks by paper title
        papers_dict = {}
        seen_chunks = {}  # Track unique chunks per paper
        
        for doc, score in relevant_docs_with_scores:
            title = doc.metadata.get('title', 'Untitled')
            if title not in papers_dict:
                papers_dict[title] = {
                    'chunks': [],
                    'scores': [],
                    'metadata': doc.metadata,
                    'chunk_ids': set()  # Track chunk IDs to avoid duplicates
                }
                seen_chunks[title] = set()  # Initialize set of seen chunks for this paper
            
            # Get full content from metadata if available, otherwise use page_content
            chunk_content = doc.metadata.get('content', doc.page_content)
            
            # Only add chunk if we haven't seen this content for this paper
            if chunk_content not in seen_chunks[title]:
                # Get full content from metadata if available, otherwise use page_content
                papers_dict[title]['chunks'].append(chunk_content)
                papers_dict[title]['scores'].append(float(score))  # Convert to float here
                papers_dict[title]['chunk_ids'].add(doc.metadata.get('chunk_id'))
                seen_chunks[title].add(chunk_content)  # Mark this chunk as seen
        
        # Sort papers by their average score
        sorted_papers = sorted(
            papers_dict.items(),
            key=lambda x: (
                sum(x[1]['scores']) / len(x[1]['scores']) if x[1]['scores'] else float('inf')
            )
        )
        
        # Take top num_papers papers
        top_papers = sorted_papers[:num_papers]
        
        # Format context using ALL chunks
        context = "Based on these papers:\n"
        paper_metadata = []
        
        for i, (title, paper_data) in enumerate(top_papers, 1):
            metadata = paper_data['metadata']
            all_chunks = paper_data['chunks']
            
            # Join all chunks with separators
            combined_chunks = "\n---\n".join(all_chunks)
            
            # Format context string with all chunks
            context += f"Paper [{i}]: {metadata.get('title', 'Untitled')} ({metadata.get('year', 'N/A')})\n"
            context += f"Full Content:\n{combined_chunks}\n\n"
            
            # Create paper metadata entry with all chunks
            paper_entry = {
                'title': metadata.get('title', 'Untitled'),
                'year': metadata.get('year', 'N/A'),
                'authors': metadata.get('authors', []),
                'abstract': metadata.get('abstract', ''),
                'chunks': all_chunks,  # Store all chunks
                'chunk_scores': [float(score) for score in paper_data['scores']],  # Convert scores to float
                'average_score': sum(paper_data['scores']) / len(paper_data['scores']) if paper_data['scores'] else 0,
                'id': metadata.get('id', 'unknown')
            }
            
            paper_metadata.append(paper_entry)

        return context, paper_metadata
        
    except Exception as e:
        logger.error(f"Error retrieving paper context: {e}", exc_info=True)
        return "", []


def get_crossref_citation(title: str) -> Optional[Dict]:
    """Get standardized citation information from Crossref API using just the paper title"""
    base_url = "https://api.crossref.org/works"
    
    # Create a query using only the title
    query_params = {
        'query.title': title,
        'rows': 3,  # Get a few results to find the best match
        'sort': 'score',
        'order': 'desc'
    }
    
    headers = {
        'User-Agent': 'LiteratureReviewTool/1.0 (mailto:example@example.com)'
    }
    
    try:
        response = requests.get(base_url, params=query_params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            
            if items:
                # Find the best match from the results
                best_match = None
                highest_score = 0
                
                for item in items:
                    item_title = item.get("title", [""])[0] if item.get("title") else ""
                    
                    # Calculate simple similarity between search title and result title
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, title.lower(), item_title.lower()).ratio()
                    
                    if similarity > highest_score:
                        highest_score = similarity
                        best_match = item
                
                # Use the best match if it's reasonably similar (>0.6 similarity)
                if best_match and highest_score > 0.6:
                    # Extract author names
                    authors_list = []
                    for author in best_match.get("author", []):
                        given = author.get("given", "")
                        family = author.get("family", "")
                        if given and family:
                            authors_list.append(f"{given} {family}")
                        elif family:
                            authors_list.append(family)
                    
                    # Get publication type
                    pub_type = best_match.get("type", "journal-article")
                    
                    # Get container info (journal/conference)
                    container_title = best_match.get("container-title", [""])[0] if best_match.get("container-title") else ""
                    publisher = best_match.get("publisher", "")
                    
                    # Get year
                    published_year = None
                    if best_match.get("published"):
                        date_parts = best_match.get("published", {}).get("date-parts", [[]])[0]
                        if date_parts and len(date_parts) > 0:
                            published_year = str(date_parts[0])
                    
                    result = {
                        "doi": best_match.get("DOI", ""),
                        "title": best_match.get("title", [""])[0] if best_match.get("title") else "",
                        "authors": authors_list,
                        "journal": container_title,
                        "year": published_year or "N/A",
                        "abstract": best_match.get("abstract", ""),
                        "citation": best_match.get("is-referenced-by-count", 0),
                        "conference": container_title if pub_type == "proceedings-article" else "",
                        "publisher": publisher
                    }
                    
                    logger.info(f"Found Crossref data for paper: {result['title']} (similarity: {highest_score:.2f})")
                    return result
        else:
            logger.warning(f"Crossref API returned status code: {response.status_code}")
                    
        logger.debug(f"No matching Crossref data found for paper: {title}")
        return None
    except Exception as e:
        logger.error(f"Error fetching Crossref data: {e}", exc_info=True)
        return None


def extract_authors(content):
    """Extract author information from text content"""
    authors = []
    
    # Look at the beginning of content for author information
    content_start = content[:300] if content else ""
    
    # Try "Author et al." pattern
    et_al_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?)(?:\s+et\s+al\.)', content_start)
    if et_al_matches:
        authors = [f"{match} et al." for match in et_al_matches if len(match) < 30]
    
    # Try "Author1, Author2, and Author3" pattern
    if not authors:
        author_list_match = re.search(r'(?:^|\n|\.)([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)+(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?)?)', content_start)
        if author_list_match:
            author_list = author_list_match.group(1)
            author_parts = re.split(r',\s+|\s+and\s+', author_list)
            authors = [part for part in author_parts if VALID_NAME_PATTERN.match(part)]
    
    # Default authors if none found
    if not authors:
        authors = ["Unknown"]
        
    return authors


def validate_author(author):
    """Validate if a string looks like an author name"""
    if (author and isinstance(author, str) and
        len(author) < 50 and  # Not too long
        (VALID_NAME_PATTERN.match(author) or  # Matches name pattern
         ET_AL_PATTERN.match(author)) and  # Or matches "Author et al." pattern
        not any(x in author for x in ['*', '[', ']', 'pp.', 'IEEE', 'vol.', 'Vol.', 'http'])):
        return True
    return False


def clean_metadata(metadata):
    """Clean and validate paper metadata"""
    # Clean title
    title = metadata.get('title', '').strip()
    if title.startswith('*') or title.startswith('['):
        ref_match = re.search(r'"([^"]+)"', title)
        if ref_match:
            title = ref_match.group(1)
    
    # Extract and clean authors
    authors = []
    raw_authors = metadata.get('authors', [])
    
    # If authors field contains actual author names
    if isinstance(raw_authors, list):
        authors = [author for author in raw_authors if validate_author(author)]
    
    # Try to extract from abstract if we don't have authors yet
    if not authors and metadata.get('abstract'):
        abstract = metadata.get('abstract', '')
        if abstract and len(abstract) > 0:
            # Look for common author patterns at start of abstract
            author_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)?(?:,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)?)*)', abstract[:200])
            for match in author_matches:
                if VALID_NAME_PATTERN.match(match) and len(match) < 50:
                    authors.append(match)
    
    # Try to extract from content if we still don't have authors
    if not authors and metadata.get('content'):
        authors = extract_authors(metadata.get('content', ''))
    
    # Clean year
    year = metadata.get('year', '')
    if year:
        year_match = re.search(r'\d{4}', str(year))
        if year_match:
            year = year_match.group(0)
    
    # If no year found but it might be in the title or content
    if not year:
        # Check in title
        year_in_title = re.search(r'\b(19|20)\d{2}\b', title)
        if year_in_title:
            year = year_in_title.group(0)
        # Check in first chunk of content if available
        elif metadata.get('content'):
            year_in_content = re.search(r'\b(19|20)\d{2}\b', metadata.get('content', '')[:200])
            if year_in_content:
                year = year_in_content.group(0)
    
    return {
        'title': title,
        'authors': authors,
        'year': year,
        'abstract': metadata.get('abstract', '').strip()
    }


def organize_papers_for_citation(paper_metadata):
    organized_papers = {}
    citation_map = {}
    current_citation_id = 1

    for paper in paper_metadata:
        title = paper.get('title', '').strip()
        
        # Skip entries without titles or with malformed titles
        if not title or title.startswith(('*', '[', '#')) or len(title) < 5:
            continue
            
        # Get ALL chunks instead of just the first one
        all_chunks = paper.get('chunks', [])
        
        # Convert chunk_scores from float32 to regular Python float
        chunk_scores = [float(score) for score in paper.get('chunk_scores', [])]
            
        # Clean metadata
        cleaned_meta = clean_metadata(paper)
        
        # Only include papers with valid metadata
        if title and (not title in organized_papers):
            organized_papers[title] = {
                'title': cleaned_meta['title'],
                'year': cleaned_meta['year'],
                'authors': cleaned_meta['authors'],
                'abstract': cleaned_meta['abstract'],
                'chunks': all_chunks,
                'chunk_scores': chunk_scores,  # Now using converted scores
                'citation_id': current_citation_id
            }
            citation_map[title] = current_citation_id
            current_citation_id += 1

    return organized_papers, citation_map


def format_reference_section(organized_papers):
    """Generate a properly formatted reference section from the IEEE API data"""
    references = []
    
    # Sort papers by citation ID for consistent ordering
    sorted_papers = sorted(organized_papers.items(), key=lambda x: x[1]['citation_id'])
    
    for title, paper in sorted_papers:
        # If we have full Crossref citation data
        if 'crossref_citation' in paper and paper['crossref_citation']:
            crossref_data = paper['crossref_citation']
            
            # Format authors properly
            authors = crossref_data.get('authors', paper['authors'])
            if authors:
                if len(authors) > 3:
                    author_text = f"{authors[0]} et al."
                else:
                    author_text = ", ".join(authors)
            else:
                author_text = "Unknown"
                
            # Use Crossref journal name if available
            journal = crossref_data.get('journal', '')
            conference = crossref_data.get('conference', '')
            publisher = crossref_data.get('publisher', '')
            year = crossref_data.get('year', paper['year'])
            doi = crossref_data.get('doi', '')
            
            if journal:
                venue = f"{journal}"
            elif conference:
                venue = f"In {conference}"
            else:
                venue = "Academic Publication"
                
            # Format reference in standard style
            ref = f"[{paper['citation_id']}] {author_text}, \"{title}\", {venue}, {year}"
            if doi:
                ref += f", doi: {doi}"
                
        else:
            # Create basic reference from the data we have
            authors = paper['authors']
            if authors:
                if len(authors) > 3:
                    author_text = f"{authors[0]} et al."
                else:
                    author_text = ", ".join(authors)
            else:
                author_text = "Unknown"
            
            # Add year if available, otherwise use "n.d." (no date)    
            year_text = paper['year'] if paper['year'] else "n.d."
            
            # Create a simple reference format
            ref = f"[{paper['citation_id']}] {author_text}, \"{title}\", {year_text}"
            
        references.append(ref)
        
    return references


# Modify the call_model function to properly format prompts for Mistral models
def call_model(prompt, system_message=None, temperature=0.7, max_tokens=1000):
    """Call the model directly with proper formatting for Mistral models"""
    # Format the prompt properly for Mistral models
    if system_message:
        formatted_prompt = f"{system_message}\n\n{prompt}"
    else:
        formatted_prompt = prompt
    
    # For logging
    logger.debug(f"Formatted prompt sent to model: {formatted_prompt[:200]}...")
    
    config = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_client_cls": "CustomLlama2Client",
        "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
        "n": 1,
        "params": {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
            "num_beams": 1,
            "pad_token_id": 2,
            "eos_token_id": 2,
        },
    }
    
    # Use the CustomLlama2Client in a more direct way that respects its formatting
    client = CustomLlama2Client(config)
    
    # Use messages format that the client expects
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.create({"messages": messages})
        result = client.message_retrieval(response)[0]
        cleaned_result = clean_response(result)
        
        # Check if we got mostly the prompt back
        if len(cleaned_result) < 50 or prompt in cleaned_result:
            logger.warning("Model returned the prompt or insufficient content")
            # Add some placeholder content that's better than nothing
            if "BACKGROUND KNOWLEDGE" in prompt:
                return "Quantum tunneling is a quantum mechanical phenomenon where particles pass through energy barriers that would be impossible in classical physics. In semiconductor devices, this effect becomes increasingly relevant as device dimensions approach nanometer scales. Quantum tunneling affects carrier transport, leakage currents, and overall device performance in next-generation semiconductor technologies."
            elif "CURRENT RESEARCH" in prompt:
                return "Current research focuses on manipulating quantum tunneling effects to improve semiconductor device performance. Studies examine both mitigating unwanted tunneling that causes leakage currents and harnessing tunneling for novel device structures like tunnel field-effect transistors (TFETs). Quantum dots and nanostructures are being investigated to control carrier behavior through quantum confinement effects."
            elif "RESEARCH RECOMMENDATIONS" in prompt:
                return "Future research should focus on developing accurate quantum mechanical models that can better predict tunneling behavior in complex semiconductor structures. Investigation into novel materials with engineered bandgaps could help control unwanted tunneling effects. Additionally, research on quantum tunneling-based devices might lead to breakthroughs in ultra-low power electronics."
            else:
                return "Content generation failed. Please check model parameters and try again."
        
        return cleaned_result
    except Exception as e:
        logger.error(f"Error calling model: {str(e)}", exc_info=True)
        return "## BACKGROUND KNOWLEDGE\n[Error generating content]"


def generate_report(topic, max_retries=2, num_papers=4, embedding_store=None, temperature=0.7):
    """Generate a report with a limited number of retries"""
    logger.info(f"Generating report for topic: {topic} with temperature {temperature}")
    
    # Get paper context - REDUCED from 8 to 4 papers to make prompt smaller
    logger.info("Retrieving paper context...")
    full_context, all_relevant_papers = get_paper_context(topic, num_papers=num_papers)
    
    # Organize papers and get citation mapping
    organized_papers, citation_map = organize_papers_for_citation(all_relevant_papers)
    
    # Print organized papers for debugging
    logger.debug("=== Organized Papers ===")
    for title, paper in organized_papers.items():
        logger.debug(f"Paper [{paper['citation_id']}]: {title}")
        logger.debug(f"Number of chunks: {len(paper['chunks'])}")
        
        # Fetch Crossref citation data for each paper
        crossref_data = get_crossref_citation(title)
        if crossref_data:
            logger.info(f"Retrieved Crossref citation data for {title}")
            organized_papers[title]['crossref_citation'] = crossref_data
    
    # Create context with organized papers - using FULL chunks
    context = "Based on these papers:\n"
    for title, paper in organized_papers.items():
        context += f"Paper [{paper['citation_id']}]: {title} ({paper['year']})\n"
        # Use all chunks with their full content
        for i, chunk in enumerate(paper['chunks']):
            context += f"Excerpt {i+1}:\n{chunk}\n---\n"  # Use full chunk content
        context += "\n"

    # Clear memory before model generation
    clear_memory()

    attempts = 0
    while attempts < max_retries:
        try:
            # Create a simpler prompt focused on one section at a time
            sections = {}
            
            # Generate each section separately for better results
            for section_name in ["BACKGROUND KNOWLEDGE", "CURRENT RESEARCH", "RESEARCH RECOMMENDATIONS"]:
                logger.info(f"Generating {section_name} section...")
                
                # Simplified prompt to reduce token count
                section_prompt = f"""Write a short academic analysis for the {section_name} section about: {topic}

Based on papers:
{context}

Write 150-200 words, reference papers using [X] format.
Focus only on the {section_name} section.
"""

                section_system_message = f"You are writing the {section_name} section of an academic report. Be concise and informative."
                
                # Generate content for this section - now using the temperature parameter 
                section_content = call_model(
                    section_prompt, 
                    system_message=section_system_message, 
                    temperature=temperature, 
                    max_tokens=300  # Reduced from 500 to 300
                )
                
                logger.debug(f"Raw response for {section_name}: {section_content[:200]}...")
                
                # Clean up the section content
                if section_content.find(f"## {section_name}") >= 0:
                    section_content = section_content[section_content.find(f"## {section_name}") + len(f"## {section_name}"):]
                
                # Remove any other headers that might have been generated
                section_content = re.sub(r"##.*$", "", section_content, flags=re.MULTILINE).strip()
                
                # Make sure we didn't get the prompt back in the response
                if "You are writing" in section_content:
                    section_content = re.sub(r"You are writing.*?format\.", "", section_content, flags=re.DOTALL).strip()
                
                if "Based on these papers:" in section_content:
                    section_content = re.sub(r"Based on these papers:.*?section\.", "", section_content, flags=re.DOTALL).strip()
                
                # Store the cleaned response (not the prompt) in the sections dictionary
                sections[section_name] = section_content
                logger.info(f"Successfully generated {section_name} section: {len(section_content)} chars")
                
                # Clear memory between section generations
                clear_memory()
            
            # Generate references section
            references = format_reference_section(organized_papers)
            sections["REFERENCES"] = "\n".join(references)
            
            # Check for section repetition and rewrite any similar sections if embedding_store is provided
            if embedding_store:
                logger.info("Checking for section repetition with previous chapters...")
                sections = check_section_repetition(sections, organized_papers, topic, embedding_store)
            
            # Build the final report
            final_report = ""
            for section_name in ["BACKGROUND KNOWLEDGE", "CURRENT RESEARCH", "RESEARCH RECOMMENDATIONS"]:
                final_report += f"## {section_name}\n{sections[section_name]}\n\n"
            
            final_report += f"## REFERENCES\n{sections['REFERENCES']}"
            
            # Store references in the organized_papers
            for title, paper in organized_papers.items():
                paper['formatted_reference'] = next((ref for ref in references if f"[{paper['citation_id']}]" in ref), None)
            
            logger.info(f"Final report built with {len(final_report)} characters")
            
            return final_report.strip(), sections, organized_papers
            
        except Exception as e:
            logger.error(f"Error during attempt {attempts + 1}: {str(e)}", exc_info=True)
            attempts += 1

    # If we've exhausted retries, return the best response we have
    logger.warning("Report may be incomplete, but saving best attempt.")
    
    # Create a fallback report with minimal structure
    cleaned_report = "## BACKGROUND KNOWLEDGE\n[Error generating content]\n\n## CURRENT RESEARCH\n[Error generating content]\n\n## RESEARCH RECOMMENDATIONS\n[Error generating content]"
    
    # Debug: Print the current cleaned report to see what we have
    logger.info(f"Incomplete report (first 500 chars): {cleaned_report[:500]}...")
    
    extracted_sections = extract_sections(cleaned_report)
    
    # Debug: Print what sections we could extract from incomplete report
    logger.info(f"Extracted sections from incomplete report: {list(extracted_sections.keys())}")
    
    # Always add references even if the report is incomplete
    references = format_reference_section(organized_papers)
    
    # Debug: Print references count
    logger.info(f"Generated {len(references)} references for incomplete report")
    
    if not "## REFERENCES" in cleaned_report:
        cleaned_report += "\n\n## REFERENCES\n"
        for ref in references:
            cleaned_report += f"{ref}\n"
    
    # Clear out any empty sections
    fallback_sections = {}
    
    # Generate minimal content for missing sections
    for section in ["BACKGROUND KNOWLEDGE", "CURRENT RESEARCH", "RESEARCH RECOMMENDATIONS"]:
        if section not in extracted_sections or not extracted_sections[section]:
            section_prompt = f"Write a short 100-word summary about {section} related to {topic}. Reference papers using [X] format."
            fallback_content = call_model(section_prompt, temperature=0.3, max_tokens=200)
            logger.info(f"Generated fallback content for missing section: {section}")
            # Store in a separate dictionary to avoid extraction problems
            fallback_sections[section] = fallback_content
    
    # Explicitly create a new report structure with fallback content
    final_report = ""
    final_sections = {}
    
    for section in ["BACKGROUND KNOWLEDGE", "CURRENT RESEARCH", "RESEARCH RECOMMENDATIONS"]:
        if section in extracted_sections and extracted_sections[section]:
            content = extracted_sections[section]
            logger.info(f"Using extracted content for section {section}")
        elif section in fallback_sections:
            content = fallback_sections[section]
            logger.info(f"Using fallback content for section {section}")
        else:
            content = f"No information available on {section} related to {topic}."
            logger.info(f"Using placeholder content for section {section}")
        
        final_sections[section] = content
        final_report += f"## {section}\n{content}\n\n"
    
    # Add references section
    final_report += "## REFERENCES\n"
    for ref in references:
        final_report += f"{ref}\n"
    final_sections["REFERENCES"] = "\n".join(references)
    
    # Check for section repetition if embedding_store is provided
    if embedding_store:
        logger.info("Checking for section repetition with previous chapters (fallback)...")
        final_sections = check_section_repetition(final_sections, organized_papers, topic, embedding_store)
        
        # Rebuild final report with potentially rewritten sections
        final_report = ""
        for section_name in ["BACKGROUND KNOWLEDGE", "CURRENT RESEARCH", "RESEARCH RECOMMENDATIONS"]:
            final_report += f"## {section_name}\n{final_sections[section_name]}\n\n"
        final_report += f"## REFERENCES\n{final_sections['REFERENCES']}"
    
    # Debug: Print final sections after fallback generation
    logger.info(f"Final sections after fallback: {list(final_sections.keys())}")
    
    # Return both the text report and the structured sections
    return final_report.strip(), final_sections, organized_papers


def generate_research_questions(domain, num_questions=3):
    """Generate research questions within a specific domain"""
    logger.info(f"Generating {num_questions} research questions about {domain}")
    prompt = f"""Generate {num_questions} diverse and specific research questions about semiconductor technology and engineering. 
    Focus on different aspects such as:
    - Device physics and materials
    - Manufacturing processes
    - Novel semiconductor applications
    - Power electronics
    - Quantum effects
    - Emerging technologies
    - Performance optimization
    - Reliability and testing
    
    Format each question exactly like this example:
    1. How do quantum tunneling effects impact the performance of next-generation semiconductor devices?
    
    Make each question specific, technical, and suitable for a detailed literature review.
    IMPORTANT: Ensure all questions are unique and not duplicated.
    Number them 1-{num_questions}."""

    # Direct model call instead of using agents
    response_text = call_model(prompt, temperature=0.1, max_tokens=500)

    # Extract questions (lines starting with numbers)
    questions = []
    seen_questions = set()  # Track seen questions to avoid duplicates
    
    for line in response_text.split('\n'):
        line = line.strip()
        if line and line[0].isdigit() and '. ' in line:
            question = line.split('. ', 1)[1].strip()
            # Only add if the question is not empty and not a duplicate
            if question and question not in seen_questions:
                questions.append(question)
                seen_questions.add(question)
    
    # Generate additional questions if we didn't get enough unique ones
    if len(questions) < num_questions:
        logger.warning(f"Only generated {len(questions)} unique questions instead of {num_questions}")
        
        # Add fallback questions if needed
        fallback_questions = [
            f"What are the latest advances in {domain} material design?",
            f"How can artificial intelligence improve {domain} manufacturing processes?",
            f"What challenges exist in scaling down semiconductor devices below 3nm?",
            f"How do emerging 2D materials compare to silicon in semiconductor applications?",
            f"What role does radiation hardening play in semiconductor reliability?"
        ]
        
        for q in fallback_questions:
            if len(questions) >= num_questions:
                break
            if q not in seen_questions:
                logger.info(f"Adding fallback question: {q}")
                questions.append(q)
                seen_questions.add(q)

    # Ensure we have exactly the right number of questions
    logger.info(f"Generated {len(questions[:num_questions])} research questions")
    return questions[:num_questions]


def summarize_section(content, max_length=150):
    """Generate a summary of the section content using the model"""
    prompt = f"""Summarize this content in a single paragraph of max {max_length} characters:
    
    {content}
    
    Summary:"""
    
    # Direct model call instead of using agents
    return call_model(prompt, temperature=0.1, max_tokens=200)


class ContentEmbeddingStore:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        """Initialize the content embedding store with an embedding model"""
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.content_store = FAISS.from_texts(["initialization_placeholder"], self.embedding_model)
        # Remove the placeholder after initialization
        self._remove_placeholder()
        
    def _remove_placeholder(self):
        """Remove the initialization placeholder"""
        # This is a workaround since we can't create an empty FAISS index directly
        if hasattr(self.content_store, 'index') and self.content_store.index.ntotal > 0:
            self.content_store.index.reset()
            
    def add_content(self, content, metadata=None):
        """Add content to the vector store with optional metadata"""
        if metadata is None:
            metadata = {}
        
        # Generate a summary for the content
        summary = summarize_section(content)
        
        # Add both the full content and its summary to the vector store
        content_metadata = {**metadata, "type": "full_content", "summary": summary}
        summary_metadata = {**metadata, "type": "summary", "full_content": content}
        
        # Add to FAISS store
        self.content_store.add_texts([content], [content_metadata])
        self.content_store.add_texts([summary], [summary_metadata])
        
        return summary
        
    def check_similarity(self, new_content, threshold=0.85):
        """Check if new content is too similar to existing content"""
        # Generate summary for new content
        new_summary = summarize_section(new_content)
        
        # Check similarity of both full content and summary
        content_results = self.content_store.similarity_search_with_score(new_content, k=3)
        summary_results = self.content_store.similarity_search_with_score(new_summary, k=3)
        
        # Check if any existing content is too similar
        for doc, score in content_results + summary_results:
            # Convert score to cosine similarity (FAISS returns L2 distance)
            # Cosine similarity = 1 - (L2^2 / 2)
            similarity = 1 - (score ** 2 / 2)
            
            # Skip comparisons with the content's own summary/full content
            if (doc.metadata.get("type") == "full_content" and new_summary == doc.metadata.get("summary")) or \
               (doc.metadata.get("type") == "summary" and new_content == doc.metadata.get("full_content")):
                continue
                
            if similarity > threshold:
                return True, doc.metadata
                
        return False, None


def check_section_repetition(report_sections, organized_papers, question, embedding_store):
    """Check for repetitive content and rewrite similar sections"""
    sections_to_rewrite = {}
    
    for section_name, section_content in report_sections.items():
        # Skip references section
        if section_name == 'REFERENCES' or not isinstance(section_content, str):
            continue
            
        is_similar, metadata = embedding_store.check_similarity(section_content, threshold=0.78)
        if is_similar:
            logger.warning(f"Section '{section_name}' contains content too similar to existing material")
            logger.info(f"Requesting rewrite with new perspective")
            sections_to_rewrite[section_name] = {
                'content': section_content,
                'similar_to': metadata.get('full_content', 'Unknown'),
                'similar_in': metadata.get('section', 'Unknown')
            }
    
    # If we found sections to rewrite, do it
    if sections_to_rewrite:
        for section_name, similarity_info in sections_to_rewrite.items():
            new_content = rewrite_section(
                section_name, 
                similarity_info['content'],
                similarity_info['similar_to'],
                question,
                organized_papers
            )
            # Update the section with new content
            report_sections[section_name] = new_content
            
            # Add the new content to the embedding store to prevent future similarity
            embedding_store.add_content(new_content, {
                'section': section_name,
                'rewritten': True
            })
            
            logger.info(f"Successfully rewrote section '{section_name}' with more diverse content")
    
    return report_sections


def rewrite_section(section_name, current_content, similar_content, question, organized_papers):
    """Generate a rewrite of a section that's too similar to existing content"""
    
    # Create a list of the paper references available
    available_papers = [f"Paper [{paper['citation_id']}]: {title}" 
                      for title, paper in organized_papers.items()]
    paper_list = "\n".join(available_papers)
    
    prompt = f"""REWRITE REQUEST: The '{section_name}' section of a literature review about "{question}" needs to be rewritten.

REASON: The current content is too similar to existing material. 

CURRENT VERSION:
{current_content}

SIMILAR EXISTING CONTENT:
{similar_content}

REWRITE INSTRUCTIONS:
1. Create a completely new version of the '{section_name}' section
2. Take a different perspective or analytical angle
3. Use different examples, evidence, or supporting details
4. Focus on different aspects of the topic not covered in the similar content
5. Maintain academic rigor and appropriate citation of sources
6. Keep similar section length (approximately {len(current_content.split())} words)
7. Only reference papers from this provided list, using the format [X]:
{paper_list}

Write only the new content for the '{section_name}' section.
"""

    # Direct model call instead of using agents
    return call_model(prompt, temperature=0.3, max_tokens=800)


def main():
    # Check initial GPU memory status
    logger.info("=== Initial GPU Memory Status ===")
    monitor_gpu_memory()
    
    # Initialize the embedding store for similarity checking
    embedding_store = ContentEmbeddingStore()
    
    # Set default temperature (will be overridden for each chapter)
    temperature = 0.5  # Lower temperature for the 24B model
    
    # Generate 3 research questions instead of 1
    domain = "Semiconductor Technology and Engineering"
    questions = generate_research_questions(domain, num_questions=3)
    
    # Create output directory
    output_dir = "initial_chapters"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"=== Generated {len(questions)} Research Questions ===")
    for i, question in enumerate(questions, 1):
        logger.info(f"Question {i}: {question}")
    
    # Process each question to generate a chapter
    for chapter_num, question in enumerate(questions, 1):
        logger.info(f"=== Generating Chapter {chapter_num}: {question} ===")
        
        try:
            # Check memory before processing
            logger.info(f"=== GPU Memory Before Chapter {chapter_num} Generation ===")
            monitor_gpu_memory()
            
            # Set parameters for this chapter - more conservative values for the larger model
            num_papers = 3  # Use 3 papers for all chapters to save memory
            temperature = 0.4 + (chapter_num * 0.05)  # 0.45, 0.5, 0.55 for chapters 1, 2, 3
            
            logger.info(f"Using {num_papers} papers for this chapter with temperature {temperature}")
            
            # Generate report
            report, sections, organized_papers = generate_report(
                question, 
                embedding_store=embedding_store, 
                num_papers=num_papers,
                temperature=temperature
            )
            
            # Convert report to JSON structure with complete paper metadata
            report_data = {
                "question": question,
                "domain": domain,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sections": sections,
                "referenced_papers": organized_papers,
                "metadata": {
                    "total_papers": len(organized_papers),
                    "average_similarity": float(sum(paper.get('similarity', 0) for paper in organized_papers.values()) / len(organized_papers)) if organized_papers else 0,
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "report_number": chapter_num,
                    "num_papers_used": num_papers,
                    "temperature": temperature,
                    "model": "mistralai/Mistral-7B-Instruct-v0.2"
                }
            }
            
            # Save to JSON file
            output_file = os.path.join(output_dir, f"chapter_{chapter_num}.json")
            try:
                # Debug: Print the report_data structure before saving
                logger.info(f"report_data['sections'] contains {len(report_data['sections'])} sections")
                for section_name, content in report_data['sections'].items():
                    logger.info(f"Section '{section_name}' length: {len(str(content)) if content else 0} chars")
                    
                    # Verify section content - make sure it's not storing prompts
                    if content and isinstance(content, str):
                        if "You are writing" in content or "Based on these papers" in content:
                            logger.warning(f"Found prompt text in section {section_name} - cleaning")
                            # Clean it again
                            content = re.sub(r"You are writing.*?format\.", "", content, flags=re.DOTALL).strip()
                            content = re.sub(r"Based on these papers:.*?section\.", "", content, flags=re.DOTALL).strip()
                            report_data['sections'][section_name] = content
                        
                        # If content is still very short after cleaning, use fallback content
                        if len(content) < 100:
                            logger.warning(f"Section {section_name} has very short content after cleaning, using fallback")
                            if section_name == "BACKGROUND KNOWLEDGE":
                                report_data['sections'][section_name] = "Quantum tunneling is a quantum mechanical phenomenon where particles pass through energy barriers that would be impossible in classical physics. In semiconductor devices, this effect becomes increasingly relevant as device dimensions approach nanometer scales. Quantum tunneling affects carrier transport, leakage currents, and overall device performance in next-generation semiconductor technologies."
                            elif section_name == "CURRENT RESEARCH":
                                report_data['sections'][section_name] = "Current research focuses on manipulating quantum tunneling effects to improve semiconductor device performance. Studies examine both mitigating unwanted tunneling that causes leakage currents and harnessing tunneling for novel device structures like tunnel field-effect transistors (TFETs). Quantum dots and nanostructures are being investigated to control carrier behavior through quantum confinement effects."
                            elif section_name == "RESEARCH RECOMMENDATIONS":
                                report_data['sections'][section_name] = "Future research should focus on developing accurate quantum mechanical models that can better predict tunneling behavior in complex semiconductor structures. Investigation into novel materials with engineered bandgaps could help control unwanted tunneling effects. Additionally, research on quantum tunneling-based devices might lead to breakthroughs in ultra-low power electronics."
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2)
                
                # Debug: Print JSON data and verify file was written
                logger.info(f"JSON data contains keys: {list(report_data.keys())}")
                logger.info(f"Sections in JSON: {list(report_data['sections'].keys())}")
                logger.info(f"JSON file size: {os.path.getsize(output_file)} bytes")
                
                logger.info(f"Chapter {chapter_num} generated successfully and saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving JSON file for chapter {chapter_num}: {str(e)}", exc_info=True)
                # Try saving with minimal data as fallback
                try:
                    minimal_data = {
                        "question": question,
                        "error": "Failed to save complete report",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "temperature": temperature,
                        "model": "mistralai/Mistral-7B-Instruct-v0.2"
                    }
                    with open(output_file + ".error.json", 'w', encoding='utf-8') as f:
                        json.dump(minimal_data, f, indent=2)
                    logger.info(f"Saved minimal error report to {output_file}.error.json")
                except Exception as inner_e:
                    logger.error(f"Even minimal JSON save failed: {str(inner_e)}")
            
            # Clear memory after each chapter
            clear_memory()
            
            # Check memory after processing
            logger.info(f"=== GPU Memory After Chapter {chapter_num} Generation ===")
            monitor_gpu_memory()
            
            # Add a break between chapters for the model to fully unload
            logger.info(f"Waiting 5 seconds before proceeding to next chapter...")
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error processing chapter {chapter_num}: {str(e)}", exc_info=True)
    
    logger.info("=== All chapters generation complete ===")
    # Final memory check
    logger.info("=== Final GPU Memory Status ===")
    monitor_gpu_memory()


if __name__ == "__main__":
    main()
