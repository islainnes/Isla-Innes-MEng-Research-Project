import time
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
from autogen import config_list_from_json, AssistantAgent, UserProxyAgent, Cache
from functools import lru_cache
import gc
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
from huggingface_hub import login
from dotenv import load_dotenv
import requests
from typing import Dict, Optional
import re

# Clear CUDA cache and set PyTorch memory management
torch.cuda.empty_cache()
torch.backends.cuda.max_memory_split_size = 512 * 1024 * 1024  # 512MB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'

# Set up model configuration
os.environ["OAI_CONFIG_LIST"] = json.dumps([
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_client_cls": "CustomLlama2Client",
        "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
        "n": 1,
        "params": {
            "max_new_tokens": 500,  # Reduced from 1000
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": 2,
            "eos_token_id": 2,
        },
    }
])

# Load environment variables
load_dotenv()
hf_token = os.environ.get('HUGGINGFACE_TOKEN')
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
login(token=hf_token)

# Add IEEE API key
os.environ['IEEE_API_KEY'] = 'cp8nvrf6ft5yh9d2a67r4cje'

# Create a temporary directory for cache
cache_dir = os.path.join(tempfile.gettempdir(), 'autogen_cache')
os.makedirs(cache_dir, exist_ok=True)

# Configure cache
config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model_client_cls": ["CustomLlama2Client"]}
)


@lru_cache(maxsize=1)
def load_shared_model(model_name, device):
    """Load model once and cache it"""
    print(f"Loading model {model_name} to {device}...")
    
    # Configure quantization settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Increase memory allocation for GPU since we have 80GB available
    max_memory = {
        0: "70GiB",  # Use up to 70GB of GPU memory
        "cpu": "24GiB"  # Keep some CPU memory available for offloading if needed
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # Let the model decide optimal device mapping
        quantization_config=quantization_config,
        max_memory=max_memory,
        offload_folder="offload"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class CustomLlama2Client:
    def __init__(self, config, **kwargs):
        self.config = config
        self.model_name = config["model"]
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.gen_params = config.get("params", {})
        self.model, self.tokenizer = load_shared_model(
            self.model_name, self.device)

    def _format_chat_prompt(self, messages):
        formatted_prompt = "<s>[INST] "
        system_message = next((m["content"]
                              for m in messages if m["role"] == "system"), None)
        if system_message:
            formatted_prompt += f"{system_message}\n\n"
        for message in messages:
            if message["role"] == "user":
                formatted_prompt += f"{message['content']}"
            elif message["role"] == "assistant":
                formatted_prompt += f" [/INST] {message['content']} </s><s>[INST] "
        formatted_prompt += " [/INST]"
        return formatted_prompt

    def create(self, params):
        response = SimpleNamespace()
        prompt = self._format_chat_prompt(params["messages"])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        response.choices = []
        response.model = self.model_name

        try:
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
            print(f"Error during generation: {str(e)}")
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
    # Split on ## INTRODUCTION and keep only what comes after
    if "## INTRODUCTION" in text:
        parts = text.split("## INTRODUCTION")
        text = "## INTRODUCTION" + parts[-1]

    text = text.replace("[/INST]", "").replace("[INST]", "")

    # Remove the references section cleanup to keep references
    return text.strip()


def extract_sections(text):
    """Extract sections from the report text into a dictionary"""
    sections = {}
    current_section = None
    current_content = []
    references = []
    
    # Regular sections processing
    for line in text.split('\n'):
        if line.startswith('## '):
            # Save previous section content
            if current_section and current_section != 'REFERENCES':
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line.replace('## ', '').strip()
            current_content = []
        else:
            # Process references differently
            if current_section == 'REFERENCES':
                # Look for reference citations like [1], [2], etc.
                if line.strip() and '[' in line:
                    references.append({
                        'id': len(references) + 1,
                        'content': line.strip()
                    })
            else:
                current_content.append(line)

    # Add the last section
    if current_section and current_section != 'REFERENCES':
        sections[current_section] = '\n'.join(current_content).strip()
    
    # Add references with IDs to sections
    if references:
        sections['REFERENCES'] = references

    return sections


def load_vector_store():
    """Load FAISS vector store using LangChain"""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError("FAISS index not found. Please create it first.")


def get_paper_context(topic, num_papers=12):
    """Get context from relevant papers using LangChain FAISS"""
    try:
        vector_store = load_vector_store()
        # First get more documents than we need to ensure diversity
        relevant_docs = vector_store.similarity_search(topic, k=num_papers * 3)
        
        # Create a dictionary to group chunks by paper title
        papers_dict = {}
        for doc in relevant_docs:
            title = doc.metadata.get('title', 'Untitled')
            if title not in papers_dict:
                papers_dict[title] = doc
        
        # Take the first num_papers unique papers
        unique_papers = list(papers_dict.values())[:num_papers]
        
        print("\n=== Retrieved Documents ===")
        for i, doc in enumerate(unique_papers, 1):
            print(f"\nDocument {i}:")
            print(f"Title: {doc.metadata.get('title', 'Untitled')}")
            print(f"Content length: {len(doc.page_content)} characters")
            print("First 200 characters of content:")
            print(doc.page_content[:200], "...\n")
            print("-" * 80)
        
        # Format context
        context = "Based on these relevant papers:\n"
        paper_metadata = []
        
        for i, doc in enumerate(unique_papers, 1):
            # Extract metadata from document
            metadata = doc.metadata
            content = doc.page_content
            
            # Format context string without truncation
            context += f"{i}. {metadata.get('title', 'Untitled')} ({metadata.get('year', 'N/A')})\n"
            context += f"   Excerpt: {content}\n\n"

            paper_metadata.append({
                'title': metadata.get('title', 'Untitled'),
                'year': metadata.get('year', 'N/A'),
                'authors': metadata.get('authors', []),
                'abstract': metadata.get('abstract', ''),
                'content': content,
                'excerpt': content,
                'similarity': metadata.get('similarity', 0.0),
                'id': metadata.get('id', 'unknown')
            })

        print("\n=== Final Context Used in Prompt ===")
        print(context[:500], "...\n")
        
        return context, paper_metadata
        
    except Exception as e:
        print(f"Error retrieving paper context: {e}")
        return "", []


def get_ieee_citation(title: str, authors: list, year: str) -> Optional[Dict]:
    """Get standardized citation information from IEEE Xplore API"""
    base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    api_key = "cp8nvrf6ft5yh9d2a67r4cje"
    
    # Construct URL with parameters exactly as per documentation
    query_url = f"{base_url}?apikey={api_key}&format=json&max_records=1&start_record=1&sort_order=relevance&title={requests.utils.quote(title)}"
    
    if year:
        query_url += f"&publication_year={year}"
    
    try:
        print(f"\nQuerying IEEE Xplore for paper: {title}")
        print(f"Using URL: {query_url}")
        
        response = requests.get(query_url)
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("total_records", 0) > 0:
                article = data["articles"][0]
                
                result = {
                    "doi": article.get("doi"),
                    "title": article.get("title"),
                    "authors": [author.get("full_name") for author in article.get("authors", [])],
                    "journal": article.get("publication_title"),
                    "year": article.get("publication_year"),
                    "abstract": article.get("abstract"),
                    "citation": article.get("citing_paper_count"),
                    "conference": article.get("conference_location"),
                    "publisher": "IEEE"
                }
                
                return result
            else:
                print(f"No matches found for paper: {title}")
        else:
            print(f"IEEE API request failed with status code: {response.status_code}")
            print(f"Full response: {response.text}")
            
        return None
    except Exception as e:
        print(f"Error fetching IEEE data: {e}")
        return None


def clean_metadata(metadata):
    """Clean and validate paper metadata"""
    # Clean title
    title = metadata.get('title', '').strip()
    if title.startswith('*') or title.startswith('['):
        ref_match = re.search(r'"([^"]+)"', title)
        if ref_match:
            title = ref_match.group(1)
    
    # Clean authors
    authors = metadata.get('authors', [])
    cleaned_authors = []
    for author in authors:
        # Remove entries that don't look like names
        if author and not any(x in author for x in ['*', '[', ']', 'pp.', 'IEEE']):
            cleaned_authors.append(author)
    
    # Clean year
    year = metadata.get('year', '')
    if year:
        year_match = re.search(r'\d{4}', str(year))
        if year_match:
            year = year_match.group(0)
    
    return {
        'title': title,
        'authors': cleaned_authors,
        'year': year,
        'abstract': metadata.get('abstract', '').strip()
    }


def organize_papers_for_citation(paper_metadata):
    organized_papers = {}
    citation_map = {}
    current_citation_id = 1

    for paper in paper_metadata:
        title = paper['title']
        if not title.startswith(('*', '[')):
            cleaned_meta = clean_metadata(paper)
            content = paper['content']
            
            # Only process papers that have valid content
            if not content.startswith(('*', '[')) and '*[' not in content:
                if title not in organized_papers:
                    organized_papers[title] = {
                        'title': cleaned_meta['title'],
                        'year': cleaned_meta['year'],
                        'authors': cleaned_meta['authors'],
                        'abstract': cleaned_meta['abstract'],
                        'chunks': [],
                        'citation_id': current_citation_id
                    }
                    citation_map[title] = current_citation_id
                    current_citation_id += 1
                
                organized_papers[title]['chunks'].append(content)

    # Remove any papers that ended up with no chunks
    organized_papers = {k: v for k, v in organized_papers.items() if v['chunks']}
    
    return organized_papers, citation_map


def generate_report(topic, max_retries=3, num_papers=12):
    """Generate a report with a limited number of retries"""
    print(f"\nGenerating report for topic: {topic}")
    
    # Get paper context and use 12 papers
    print("\nRetrieving paper context...")
    full_context, all_relevant_papers = get_paper_context(topic, num_papers=12)
    
    # Organize papers and get citation mapping
    organized_papers, citation_map = organize_papers_for_citation(all_relevant_papers)
    
    # Print organized papers for debugging
    print("\n=== Organized Papers ===")
    for title, paper in organized_papers.items():
        print(f"\nPaper [{paper['citation_id']}]: {title}")
        print(f"Number of chunks: {len(paper['chunks'])}")
    
    # Create context with organized papers
    context = "Based on these papers:\n"
    for title, paper in organized_papers.items():
        context += f"Paper [{paper['citation_id']}]: {title} ({paper['year']})\n"
        for chunk in paper['chunks']:
            context += f"Excerpt: {chunk}\n"
        context += "\n"

    # Update the prompt to reference available papers
    available_citations = sorted(list(set(citation_map.values())))
    citation_list = [f"Paper [{i}]" for i in available_citations]
    
    prompt = f"""Write a comprehensive academic report and literature review on: {topic}

{context}

IMPORTANT: You must ONLY use and reference the papers provided above, using their assigned numbers: {', '.join(citation_list)}

IMPORTANT GUIDELINES:
1. Total report length must be approximately 800 words.
2. Section lengths should be:
   - INTRODUCTION: ~150 words
   - METHODOLOGY: ~150 words
   - RESULTS: ~300 words
   - DISCUSSION: ~100 words
   - CONCLUSION: ~100 words

3. CRITICAL: You may ONLY reference papers using the citation format [X] where X is the paper's assigned number.
4. When multiple excerpts from the same paper are used, use the same citation number.
5. If the provided papers don't cover certain aspects, acknowledge the limitations rather than making assumptions.
6. Focus on analyzing and synthesizing the findings from all available papers.

Write the report content starting with ## INTRODUCTION and ending with ## REFERENCES."""

    writer_system_message = """You are a technical writer. Write clear, structured academic reports using markdown headers.
    Always include the following sections with ## headers:
    ## INTRODUCTION
    ## METHODOLOGY
    ## RESULTS
    ## DISCUSSION
    ## CONCLUSION
    ## REFERENCES"""

    writer = AssistantAgent(
        name="Writer",
        system_message=writer_system_message,
        llm_config={
            "config_list": config_list,
            "cache_seed": None,  # Disable caching
            "timeout": 600,
            "max_retries": 3,
            "temperature": 0.1,
        }
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1
    )

    writer.register_model_client(model_client_cls=CustomLlama2Client)

    attempts = 0
    while attempts < max_retries:
        try:
            chat_response = user_proxy.initiate_chat(
                writer, message=prompt, silent=True)
            report = user_proxy.last_message()['content']

            # Clean the response
            cleaned_report = clean_response(report)

            # Better completion check
            if "## REFERENCES" in cleaned_report:
                # Check if we have content after REFERENCES
                references_parts = cleaned_report.split("## REFERENCES")
                if len(references_parts) > 1 and references_parts[1].strip():
                    return cleaned_report, organized_papers

            print(f"\nAttempt {attempts + 1} produced incomplete response. Retrying...")
            attempts += 1

        except Exception as e:
            print(f"\nError during attempt {attempts + 1}: {str(e)}")
            attempts += 1

    # If we've exhausted retries, return the best response we have
    print("\nWarning: Report may be incomplete, but saving best attempt.")
    return cleaned_report.strip(), organized_papers


def generate_research_questions(domain):
    """Generate research questions within a specific domain"""
    prompt = f"""Generate 20 diverse and specific research questions about semiconductor technology and engineering. 
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
    Number them 1-20."""

    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1
    )

    question_generator = AssistantAgent(
        name="QuestionGenerator",
        system_message="Generate numbered research questions only. No additional text.",
        llm_config={
            "config_list": config_list,
            "cache_seed": None,  # Disable caching
            "temperature": 0.1,
        }
    )

    question_generator.register_model_client(
        model_client_cls=CustomLlama2Client)

    chat_response = user_proxy.initiate_chat(
        question_generator, message=prompt, silent=True)
    response_text = user_proxy.last_message()['content'].strip()

    # Extract questions (lines starting with numbers)
    questions = []
    for line in response_text.split('\n'):
        line = line.strip()
        if line and line[0].isdigit() and '. ' in line:
            question = line.split('. ', 1)[1].strip()
            if question:
                questions.append(question)

    # Ensure we have exactly 20 questions
    if len(questions) < 20:
        print(
            f"Warning: Only generated {len(questions)} questions instead of 20")
    elif len(questions) > 20:
        questions = questions[:20]

    if not questions:
        # Fallback questions if extraction fails
        questions = [
            f"What are the latest advances in {domain}?",
            f"How can artificial intelligence improve {domain}?",
            # ... add more fallback questions ...
        ]
        print("Warning: Using fallback questions due to parsing error")

    return questions


def main():
    # First generate 20 semiconductor research questions
    domain = "Semiconductor Technology and Engineering"
    questions = generate_research_questions(domain)
    
    # Create output directory
    output_dir = "litreviews"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Generated Research Questions ===")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    # Generate reports for each question
    for i, question in enumerate(questions, 1):
        print(f"\n=== Generating Report {i}/20 ===")
        print(f"Question: {question}")
        
        try:
            # Generate report for the specific question
            report, organized_papers = generate_report(question)
            cleaned_report = clean_response(report)
            
            # Convert report to JSON structure with complete paper metadata
            report_data = {
                "question": question,
                "domain": domain,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sections": extract_sections(cleaned_report),
                "referenced_papers": organized_papers,
                "metadata": {
                    "total_papers": len(organized_papers),
                    "average_similarity": sum(paper.get('similarity', 0) for paper in organized_papers.values()) / len(organized_papers) if organized_papers else 0,
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "report_number": i
                }
            }
            
            # Save to JSON file with paper_X naming convention
            output_file = os.path.join(output_dir, f"paper_{i}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"Report {i} generated successfully and saved to {output_file}")
            
            # Clear memory after each report
            torch.cuda.empty_cache()
            gc.collect()
            
            # Add a small delay between reports to prevent rate limiting
            time.sleep(5)
            
        except Exception as e:
            print(f"Error generating report {i}: {str(e)}")
            continue
    
    print("\n=== All reports generated ===")


if __name__ == "__main__":
    main()
