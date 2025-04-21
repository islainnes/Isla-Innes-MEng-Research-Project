import time
import sys
import json
import os
import logging
import torch
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import login
import argparse
import re

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
# Setting more conservative memory splits for smaller GPUs
torch.backends.cuda.max_memory_split_size = 128 * 1024 * 1024  # 128MB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()
hf_token = os.environ.get('HUGGINGFACE_TOKEN')
if not hf_token:
    logger.critical("HUGGINGFACE_TOKEN not found in environment variables")
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
login(token=hf_token)

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
    
    # Clear CUDA cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        # Configure quantization settings with less aggressive memory optimization for large GPU
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Changed to float16 for better compatibility
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8  # Explicitly set storage type
        )
        
        # Large GPU memory settings
        max_memory = {
            0: "70GiB",  # Reduced slightly to leave more headroom
            "cpu": "32GiB"
        }
        
        logger.info(f"Using max GPU memory: {max_memory[0]}")
        
        # Set environment variables for better CUDA handling
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Changed to bfloat16
            device_map="auto",
            quantization_config=quantization_config,
            max_memory=max_memory,
            offload_folder="offload",
            trust_remote_code=True,  # Added for better model loading
            use_flash_attention_2=False,  # Explicitly disable flash attention
            attn_implementation='eager'  # Explicitly use eager implementation
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True  # Added for better model loading
        )
        
        # Ensure proper token setup
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.bos_token
                
        logger.info(f"Model and tokenizer loaded successfully")
        
        # Monitor GPU memory after loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory after loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        # Cache the model
        _model_cache[cache_key] = (model, tokenizer)
        return _model_cache[cache_key]
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

class CustomGemmaClient:
    def __init__(self, config, **kwargs):
        self.config = config
        self.model_name = "google/gemma-3-27b-it"  # Updated to use 27B model
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Remove attention parameters from gen_params if they exist
        self.gen_params = config.get("params", {}).copy()
        if "use_flash_attention_2" in self.gen_params:
            self.gen_params.pop("use_flash_attention_2")
        if "attn_implementation" in self.gen_params:
            self.gen_params.pop("attn_implementation")
            
        logger.info(f"Initializing CustomGemmaClient with model {self.model_name} on {self.device}")
        self.model, self.tokenizer = load_shared_model(
            self.model_name, self.device)

    def _format_chat_prompt(self, messages):
        formatted_prompt = ""
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
        
        if system_message:
            formatted_prompt = f"<start_of_turn>system\n{system_message}<end_of_turn>\n"
        
        for message in messages:
            if message["role"] == "user":
                formatted_prompt += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"
            elif message["role"] == "assistant" and message["role"] != "system":
                formatted_prompt += f"<start_of_turn>model\n{message['content']}<end_of_turn>\n"
        
        # Add the final model turn prompt
        formatted_prompt += "<start_of_turn>model\n"
        
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
                # Clean up the Gemma output format
                if "<start_of_turn>model\n" in generated_text:
                    generated_text = generated_text.split("<start_of_turn>model\n")[-1]
                if "<end_of_turn>" in generated_text:
                    generated_text = generated_text.split("<end_of_turn>")[0]
                
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
                # Clean up the Gemma output format
                if "<start_of_turn>model\n" in generated_text:
                    generated_text = generated_text.split("<start_of_turn>model\n")[-1]
                if "<end_of_turn>" in generated_text:
                    generated_text = generated_text.split("<end_of_turn>")[0]
                    
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
    """Remove model artifacts from generated text and extract only the actual rewritten content"""
    # Remove common prefixes from LLM outputs
    text = text.replace("<end_of_turn>", "").replace("<start_of_turn>model", "").strip()
    
    # Check if the response contains the chat format with "model" tag
    if "model\n" in text:
        # Extract everything after the last occurrence of "model\n"
        model_pos = text.rfind("model\n")
        if model_pos >= 0:
            text = text[model_pos + 6:].strip()  # +6 to account for "model\n"
    
    # If the model is just returning the prompt or system message
    if text.startswith("You are"):
        # Find the first sentence end if it exists
        end_pos = text.find("\n\n")
        if end_pos >= 0:
            text = text[end_pos+2:].strip()
    
    # Check if the response is just echoing our prompt
    if "Please rewrite the following text" in text and "IMPROVEMENT POINTS:" in text and "ORIGINAL TEXT:" in text:
        # Try to extract just the rewritten part, which should come after all the prompt text
        # Look for phrases that might signal the start of the actual rewritten content
        start_markers = [
            "REWRITTEN TEXT:",
            "Here is the rewritten version:",
            "Rewritten version:",
            "Here's my rewrite:",
            "The rewritten text:",
            "Rewritten Text:"  # Added this variation with capital T
        ]
        
        for marker in start_markers:
            if marker in text:
                start_pos = text.find(marker) + len(marker)
                text = text[start_pos:].strip()
                break
        else:
            # If no marker is found, try to find the end of the prompt
            # The prompt ends with a line asking for a rewritten version
            prompt_end_markers = [
                "Please provide a rewritten version",
                "The rewrite should be",
                "while maintaining the original"
            ]
            
            for marker in prompt_end_markers:
                if marker in text:
                    marker_pos = text.find(marker)
                    end_of_line = text.find("\n", marker_pos)
                    if end_of_line > 0:
                        text = text[end_of_line:].strip()
                        break
            
            # If still no marker found, try extracting from the end
            # Look for the response pattern where model provides rewritten text at the end
            if text.lower().find("return only the rewritten text") >= 0:
                # Find a separator like "---" or "===" often used before the actual content
                for separator in ["---", "===", "***", "\n\n"]:
                    sep_pos = text.rfind(separator)
                    if sep_pos > 0 and sep_pos > len(text) / 2:  # Only if in latter half of text
                        text = text[sep_pos + len(separator):].strip()
                        break
    
    # Check for section headings which often indicate the start of the rewritten content
    heading_match = re.search(r'^#{1,6}\s+', text)
    if heading_match:
        # If there's a heading at the start, that's likely the beginning of the actual content
        # So we keep the text as is
        pass
    # If there are multiple headings, find the first one as it's likely the start of the content
    elif '##' in text and not text.startswith('##'):
        first_heading = text.find('##')
        if first_heading > 0:
            text = text[first_heading:].strip()
    
    return text

def rewrite_text(improvement_points, original_text, temperature=0.7, max_tokens=2000, referenced_papers=None, full_chapter_context=None):
    """
    Rewrite text based on improvement points using the model.
    
    Args:
        improvement_points: List of improvement points to address
        original_text: The original text to be rewritten
        temperature: Temperature for the LLM
        max_tokens: Maximum tokens for the response
        referenced_papers: Dictionary of referenced papers for citation context
        full_chapter_context: Full chapter text for better contextual understanding
        
    Returns:
        str: The rewritten text
    """
    if not improvement_points or not original_text:
        logger.error("No improvement points or original text provided")
        return "Error: Improvement points and original text are required"
    
    # Check if it's safe to process in terms of memory
    if not estimate_memory_needs(len(original_text)):
        logger.info("Using batch processing for large text")
        rewritten_text = batch_process_text(original_text, improvement_points, max_length=5000, referenced_papers=referenced_papers, full_chapter_context=full_chapter_context)
    else:
        # Create the prompt
        prompt = create_rewrite_prompt(original_text, improvement_points, referenced_papers, full_chapter_context)
        
        # System message to guide the model
        system_message = "You are an expert academic editor specializing in technical scientific content, particularly physics and semiconductor technology. You excel at rewriting text to improve clarity, technical accuracy, and flow."
        
        # Call the model
        logger.info("Calling model to rewrite text...")
        rewritten_text = call_model(prompt, system_message=system_message, temperature=temperature, max_tokens=max_tokens)
    
    # Clean the response
    cleaned_text = clean_response(rewritten_text)
    return cleaned_text

def call_model(prompt, system_message=None, temperature=0.7, max_tokens=1000):
    """Call the model directly with proper formatting for Gemma models"""
    # Format the prompt properly for Gemma
    if system_message:
        formatted_prompt = f"{system_message}\n\n{prompt}"
    else:
        formatted_prompt = prompt
    
    # For logging
    logger.debug(f"Formatted prompt sent to model: {formatted_prompt[:200]}...")
    
    config = {
        "model": "google/gemma-2b-it",
        "model_client_cls": "CustomGemmaClient",
        "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
        "n": 1,
        "params": {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
            "num_beams": 1,
        },
    }
    
    # Use the CustomGemmaClient
    client = CustomGemmaClient(config)
    
    # Use messages format that the client expects
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    try:
        logger.info("Sending request to model...")
        response = client.create({"messages": messages})
        logger.info("Received response from model")
        result = client.message_retrieval(response)[0]
        logger.info(f"Raw response length: {len(result)} characters")
        
        # Return the raw result without any cleaning
        return result
    except Exception as e:
        logger.error(f"Error calling model: {str(e)}", exc_info=True)
        return "Error generating response"

def extract_from_json(json_file_path):
    """
    Extract improvement points and original text from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        tuple: (improvement_points, original_text, referenced_papers, full_chapter_context)
    """
    try:
        logger.info(f"Reading JSON file: {json_file_path}")
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata about the chapter/section
        metadata = data.get('metadata', {})
        chapter = metadata.get('chapter', 'Unknown')
        section = metadata.get('section', 'Unknown')
        logger.info(f"Processing Chapter {chapter}, Section: {section}")
        
        # Extract improvement points
        # Look for the first iteration that has improvement points
        improvement_points = []
        original_text = ""
        
        for iteration in data.get('iterations', []):
            if 'improvement_points' in iteration:
                improvement_points = iteration['improvement_points']
                logger.info(f"Found {len(improvement_points)} improvement points")
            
            # Extract the original text
            if 'text' in iteration and 'before' in iteration['text']:
                original_text = iteration['text']['before']
                logger.info(f"Found original text ({len(original_text)} characters)")
            
            # Once we have both, we can break
            if improvement_points and original_text:
                break
        
        # Extract referenced papers
        referenced_papers = data.get('referenced_papers', {})
        if referenced_papers:
            logger.info(f"Found {len(referenced_papers)} referenced papers")
        
        # Extract full chapter context by combining all sections
        full_chapter_context = ""
        sections = {}
        
        # Try to find sections in the data
        # Look in different locations where section content might be stored
        if 'sections' in data:
            sections = data['sections']
        elif 'original_sections' in data:
            sections = data['original_sections']
        elif 'chapter_sections' in data:
            sections = data['chapter_sections']
        
        if sections:
            # Combine all sections into a single text for context
            section_texts = []
            for section_name, section_content in sections.items():
                if isinstance(section_content, str):
                    section_texts.append(f"## {section_name}\n\n{section_content}")
                elif isinstance(section_content, dict) and 'content' in section_content:
                    section_texts.append(f"## {section_name}\n\n{section_content['content']}")
            
            if section_texts:
                full_chapter_context = "\n\n".join(section_texts)
                logger.info(f"Found full chapter context ({len(full_chapter_context)} characters)")
        
        return improvement_points, original_text, referenced_papers, full_chapter_context
    
    except Exception as e:
        logger.error(f"Error extracting from JSON: {str(e)}", exc_info=True)
        return [], "", {}, ""

def create_rewrite_prompt(original_text, improvement_points, referenced_papers=None, full_chapter_context=None):
    """
    Create a prompt for the LLM to rewrite the text based on improvement points.
    
    Args:
        original_text: The original text to be rewritten
        improvement_points: List of improvement points to address
        referenced_papers: Dictionary of referenced papers for citation context
        full_chapter_context: Full chapter text for better contextual understanding
        
    Returns:
        str: The prompt for the LLM
    """
    # Convert improvement points to a numbered list as a string
    improvement_list = "\n".join(improvement_points)
    
    # Add reference information if available
    reference_info = ""
    if referenced_papers:
        reference_info = "\nCITATION REFERENCE INFORMATION:\n"
        for citation_id, paper_info in referenced_papers.items():
            title = paper_info.get('title', 'Unknown Title')
            abstract = paper_info.get('abstract', 'No abstract available')
            reference_info += f"[{citation_id}] {title}\n"
            reference_info += f"Abstract: {abstract[:200]}...\n\n"
    
    # Add chapter context information if available
    context_info = ""
    if full_chapter_context:
        # Increased context window for 27B model
        context_info = "\nCHAPTER CONTEXT (for reference only):\n"
        context_info += f"{full_chapter_context[:2000]}...\n"
    
    prompt = f"""You will rewrite an academic text about quantum tunneling and semiconductors based on specific improvement points. Focus on making substantial, meaningful changes that address each point.

IMPROVEMENT POINTS TO ADDRESS:
{improvement_list}

ORIGINAL TEXT TO REWRITE:
{original_text}
{reference_info}
{context_info}

INSTRUCTIONS:
1. Make substantial changes that clearly address each improvement point
2. Maintain technical accuracy while improving clarity
3. Ensure the rewritten text flows well and has proper transitions
4. IMPORTANT: Preserve ALL citation markers (e.g., [1], [2], etc.) exactly as they appear in the original text
5. DO NOT remove any citations from the original text - this will break the technical accuracy of the document
6. DO NOT add any new citations that weren't in the original text
7. Return ONLY the rewritten text, not your reasoning or the original text
"""
    return prompt

def estimate_memory_needs(text_length):
    """
    Estimates memory needs based on the length of the text to process.
    
    Args:
        text_length: Length of the text in characters
        
    Returns:
        bool: True if safe to process, False if potentially too large
    """
    # Adjusted threshold for 27B model on 80GB GPU
    if text_length > 30000:  # Increased threshold for larger GPU
        logger.warning(f"Text length {text_length} may be too large even for 80GB GPU.")
        return False
    return True

def batch_process_text(original_text, improvement_points, max_length=5000, referenced_papers=None, full_chapter_context=None):
    """
    Process large text in batches if needed.
    
    Args:
        original_text: The original text to be rewritten
        improvement_points: List of improvement points
        max_length: Maximum text length to process at once
        referenced_papers: Dictionary of referenced papers
        full_chapter_context: Full chapter text
        
    Returns:
        str: The rewritten text
    """
    if len(original_text) <= max_length:
        # Process normally if text is within limits
        prompt = create_rewrite_prompt(original_text, improvement_points, referenced_papers, full_chapter_context)
        system_message = "You are an expert academic editor specializing in technical scientific content, particularly physics and semiconductor technology. You excel at rewriting text to improve clarity, technical accuracy, and flow."
        return call_model(prompt, system_message=system_message, temperature=0.7, max_tokens=1500)
    
    # If text is too long, split it (this is a simple approach - paragraphs would be better)
    logger.info(f"Text is too long ({len(original_text)} chars), processing in batches")
    sentences = original_text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence + '.' if not sentence.endswith('.') else sentence
        if current_length + len(sentence) > max_length:
            # Process current chunk
            chunk_text = ' '.join(current_chunk)
            prompt = create_rewrite_prompt(chunk_text, improvement_points, referenced_papers, full_chapter_context)
            system_message = "You are an expert academic editor specializing in technical scientific content, particularly physics and semiconductor technology. You excel at rewriting text to improve clarity, technical accuracy, and flow."
            result = call_model(prompt, system_message=system_message, temperature=0.7, max_tokens=1500)
            chunks.append(clean_response(result))
            
            # Reset for next chunk
            current_chunk = [sentence]
            current_length = len(sentence)
            
            # Clean up memory after each chunk
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    # Process the last chunk if it exists
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        prompt = create_rewrite_prompt(chunk_text, improvement_points, referenced_papers, full_chapter_context)
        system_message = "You are an expert academic editor specializing in technical scientific content, particularly physics and semiconductor technology. You excel at rewriting text to improve clarity, technical accuracy, and flow."
        result = call_model(prompt, system_message=system_message, temperature=0.7, max_tokens=1500)
        chunks.append(clean_response(result))
    
    # Combine all rewritten chunks
    return ' '.join(chunks)

def rewrite_text_from_json(json_file_path, temperature=0.7, max_tokens=2000):
    """
    Extract information from a JSON file and rewrite the text based on improvement points.
    
    Args:
        json_file_path: Path to the JSON file
        temperature: Temperature for the LLM
        max_tokens: Maximum tokens for the response
        
    Returns:
        str: The rewritten text
    """
    # Extract improvement points, original text, and additional context
    improvement_points, original_text, referenced_papers, full_chapter_context = extract_from_json(json_file_path)
    
    if not improvement_points or not original_text:
        logger.error("Failed to extract improvement points or original text from JSON")
        return "Error: Could not extract necessary information from JSON file"
    
    # Use the updated rewrite_text function with all extracted context
    return rewrite_text(
        improvement_points=improvement_points,
        original_text=original_text,
        temperature=temperature,
        max_tokens=max_tokens,
        referenced_papers=referenced_papers,
        full_chapter_context=full_chapter_context
    )

def process_folder(folder_path='outputs', temperature=0.7, max_tokens=2000):
    """
    Process all JSON files in a folder.
    
    Args:
        folder_path: Path to the folder containing JSON files
        temperature: Temperature for the LLM
        max_tokens: Maximum tokens for the response
    """
    logger.info(f"Processing all JSON files in {folder_path} folder")
    
    # Get all JSON files in the directory
    try:
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        results = {}
        
        for json_file in json_files:
            json_file_path = os.path.join(folder_path, json_file)
            logger.info(f"Processing file: {json_file}")
            
            try:
                # Call the function to rewrite text from JSON
                rewritten_text = rewrite_text_from_json(json_file_path, temperature=temperature, max_tokens=max_tokens)
                
                # Save the rewritten text to a file
                output_path = json_file_path.replace('.json', '_rewritten.txt')
                with open(output_path, 'w') as f:
                    f.write(rewritten_text)
                logger.info(f"Saved rewritten text to {output_path}")
                
                # Store results
                results[json_file] = {
                    'status': 'success',
                    'output_path': output_path
                }
                
            except Exception as e:
                logger.error(f"Error processing file {json_file}: {str(e)}", exc_info=True)
                results[json_file] = {
                    'status': 'error',
                    'error': str(e)
                }
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Error accessing directory {folder_path}: {str(e)}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

def main():
    """Main function that processes all JSON files in the outputs directory."""
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Rewrite text based on improvement points using Gemma 2B model.")
    parser.add_argument("--folder", type=str, default="outputs", help="Folder containing JSON files to process")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation (default: 0.7)")
    parser.add_argument("--max_tokens", type=int, default=1500, help="Maximum tokens in generated response (default: 1500)")
    parser.add_argument("--single_file", type=str, default=None, help="Process only a specific JSON file")
    parser.add_argument("--batch_size", type=int, default=5000, help="Maximum characters to process at once (default: 5000)")
    args = parser.parse_args()
    
    # Check initial GPU memory status
    logger.info("=== Initial GPU Memory Status ===")
    monitor_gpu_memory()
    
    # Set path to outputs directory
    outputs_dir = args.folder
    logger.info(f"Processing JSON files in {outputs_dir} directory")
    
    try:
        # Determine which files to process
        if args.single_file:
            if os.path.exists(os.path.join(outputs_dir, args.single_file)):
                json_files = [args.single_file]
                logger.info(f"Processing single file: {args.single_file}")
            else:
                logger.error(f"Specified file {args.single_file} not found in {outputs_dir}")
                return
        else:
            # Get all JSON files in the outputs directory
            json_files = [f for f in os.listdir(outputs_dir) if f.endswith('.json')]
            logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each JSON file
        for json_file in json_files:
            json_file_path = os.path.join(outputs_dir, json_file)
            logger.info(f"Processing file: {json_file}")
            
            try:
                # Extract improvement points and original text
                improvement_points, original_text, referenced_papers, full_chapter_context = extract_from_json(json_file_path)
                
                if not improvement_points or not original_text:
                    logger.warning(f"No improvement points or original text found in {json_file}")
                    continue
                
                # Display what we're working with
                logger.info(f"Original text length: {len(original_text)} characters")
                logger.info(f"Improvement points: {improvement_points}")
                
                # Use the new rewrite_text function
                cleaned_response = rewrite_text(
                    improvement_points=improvement_points,
                    original_text=original_text,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    referenced_papers=referenced_papers,
                    full_chapter_context=full_chapter_context
                )
                
                # Save the rewritten text to a file
                output_path = json_file_path.replace('.json', '_rewritten.txt')
                with open(output_path, 'w') as f:
                    f.write(cleaned_response)
                logger.info(f"Saved rewritten text to {output_path}")
                
                # Print information about the result
                print("\n" + "="*80)
                print(f"PROCESSED: {json_file}")
                print(f"OUTPUT: {output_path}")
                print(f"OUTPUT LENGTH: {len(cleaned_response)} characters")
                print("="*80 + "\n")
                
                # Monitor GPU memory after processing each file
                logger.info("=== GPU Memory Status After Processing ===")
                monitor_gpu_memory()
                
                # Force garbage collection to free up memory
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing file {json_file}: {str(e)}", exc_info=True)
                continue
        
    except Exception as e:
        logger.error(f"Error accessing outputs directory: {str(e)}", exc_info=True)
    
    logger.info("=== Processing complete ===")

if __name__ == "__main__":
    main()





