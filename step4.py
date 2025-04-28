import os
import time
import logging
import argparse
import sys
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

def setup_logging(log_level=logging.INFO):
    """Configure logging with both file and console output."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"concept_diagrams_{timestamp}.log"
    
    # Create formatters for different handlers
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Create and configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

# Initialize logger
logger = setup_logging()

# Set up GPU configurations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Setting more conservative memory splits for smaller GPUs
    torch.backends.cuda.max_memory_split_size = 128 * 1024 * 1024  # 128MB
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    logger.info("CUDA is available. GPU configurations set.")
else:
    logger.warning("CUDA is not available. Using CPU mode.")

# Model cache for reusing loaded models
_model_cache = {}

def load_model(model_name="google/gemma-3-27b-it", device="cuda"):
    """Load model once and cache it"""
    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        logger.info(f"Using cached model {model_name}")
        return _model_cache[cache_key]
    
    logger.info(f"Loading model {model_name} to {device}...")
    
    # Configure quantization settings for the larger model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # Changed to bfloat16 for better stability
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4"
    )
    
    # Adjust memory settings for the larger model
    max_memory = {
        0: "70GiB",  # Increased for the 27B model
        "cpu": "32GiB"  # Increased CPU memory for the larger model
    }
    
    logger.info(f"Using max GPU memory: {max_memory[0]}, CPU memory: {max_memory['cpu']}")
    
    try:
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
        
        # Set environment variables for better CUDA handling
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Changed to bfloat16
            device_map="auto",
            quantization_config=quantization_config,
            max_memory=max_memory,
            offload_folder="offload",
            trust_remote_code=True,  # Added for better model loading
            use_flash_attention_2=False  # Disabled flash attention
        )
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True  # Added for better tokenizer loading
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
        logger.error(f"Error loading model: {str(e)}")
        logger.error("Exception details:", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def read_markdown_file(file_path):
    """Read a markdown file and return the content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from first heading if present
        lines = content.split('\n')
        title = None
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        if not title:
            title = Path(file_path).stem.replace('_', ' ').title()
            
        # Remove any empty lines at the beginning and end
        content = content.strip()
        
        # Check if content is actually present
        if not content or len(content) < 50:  # Require at least 50 chars of content
            logger.warning(f"Insufficient content in {file_path} (only {len(content)} chars)")
            return title, None
            
        return title, content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return None, None

def generate_concept_diagram(model, tokenizer, title, content):
    """Generate a Mermaid diagram for a key concept in the background knowledge section"""
    try:
        # Truncate content if too long
        max_content_length = 10000  # Increased limit for content length with 80GB GPU
        original_length = len(content)
        if original_length > max_content_length:
            content = content[:max_content_length] + "...[content truncated]"
            logger.warning(f"Content truncated from {original_length} to {max_content_length} characters")
        
        logger.info(f"Generating concept diagram for: {title}")
        logger.debug(f"Content length: {len(content)} characters")
        
        # Extract background knowledge section if it exists
        background_section = content
        if "## BACKGROUND KNOWLEDGE" in content:
            sections = content.split("##")
            for section in sections:
                if section.strip().startswith("BACKGROUND KNOWLEDGE"):
                    background_section = section.strip()
                    break
        
        prompt = f"""
You are an expert in creating visual diagrams. Identify ONE key concept from the background knowledge that would benefit most from visual representation. Then create a focused Mermaid diagram that clearly explains this single concept.

Document title: "{title}"

Background knowledge content:
{background_section}

Guidelines:
1. First identify a single concept that is complex or central to understanding the material
2. Use valid Mermaid syntax (flowchart, mindmap, etc.) appropriate for this specific concept
3. Create a clear, focused diagram that helps explain this one concept thoroughly
4. Include only elements directly related to this single concept
5. Use an appropriate diagram type for this specific concept
6. Exclude any text outside the mermaid code block

Return ONLY the Mermaid code block, nothing else.
"""
        
        logger.debug("Preparing model inputs...")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.debug("Generating diagram with model...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Raw model response length: {len(response)} characters")
        
        # Try multiple methods to extract the Mermaid code
        mermaid_code = None
        
        # Method 1: Extract between ```mermaid and ```
        if "```mermaid" in response:
            logger.debug("Attempting extraction method 1 (```mermaid tags)")
            try:
                parts = response.split("```mermaid", 1)[1].split("```", 1)
                if parts:
                    mermaid_code = parts[0].strip()
                    logger.debug(f"Method 1 extracted mermaid code of length {len(mermaid_code)}")
            except Exception as e:
                logger.warning(f"Method 1 extraction failed: {str(e)}")
        
        # Method 2: Extract content between any ``` tags if it contains mermaid-like syntax
        if not mermaid_code and "```" in response:
            logger.debug("Attempting extraction method 2 (generic code blocks)")
            try:
                blocks = response.split("```")
                for i in range(1, len(blocks), 2):
                    block = blocks[i].strip()
                    # Check if this looks like a mermaid diagram
                    if block.startswith("mermaid"):
                        mermaid_code = block[7:].strip()  # Remove "mermaid" from the start
                        logger.debug(f"Method 2 extracted mermaid code of length {len(mermaid_code)}")
                        break
                    elif any(x in block.lower() for x in ["graph ", "flowchart ", "mindmap", "classDiagram", "sequenceDiagram"]):
                        mermaid_code = block
                        logger.debug(f"Method 2 extracted mermaid-like code of length {len(mermaid_code)}")
                        break
            except Exception as e:
                logger.warning(f"Method 2 extraction failed: {str(e)}")
        
        # Method 3: Look for diagram syntax directly
        if not mermaid_code:
            logger.debug("Attempting extraction method 3 (direct syntax search)")
            try:
                diagram_starters = ["graph ", "flowchart ", "mindmap", "classDiagram", "sequenceDiagram"]
                for starter in diagram_starters:
                    if starter in response.lower():
                        start_idx = response.lower().find(starter)
                        # Try to find end of diagram (look for typical ending patterns)
                        possible_end = response.find("```", start_idx)
                        if possible_end == -1:
                            possible_end = len(response)
                        mermaid_code = response[start_idx:possible_end].strip()
                        logger.debug(f"Method 3 extracted mermaid-like code of length {len(mermaid_code)}")
                        break
            except Exception as e:
                logger.warning(f"Method 3 extraction failed: {str(e)}")
        
        # If we found mermaid code, clean it and return it
        if mermaid_code:
            # Remove any markdown language specifier if it's still there
            if mermaid_code.startswith("mermaid\n"):
                mermaid_code = mermaid_code[8:].strip()
                logger.debug("Removed 'mermaid' prefix from code")
            
            # Verify this doesn't look like the example from the prompt
            if "root((Main Topic))" in mermaid_code and "Topic 1" in mermaid_code and "Subtopic A" in mermaid_code:
                logger.warning("Detected example diagram in response, rejecting")
                return None
                
            logger.info("Successfully extracted and cleaned mermaid diagram code")
            return f"```mermaid\n{mermaid_code}\n```"
        
        logger.warning("No valid mermaid code block found in response")
        return None
        
    except Exception as e:
        logger.error(f"Error generating diagram: {str(e)}")
        logger.error("Exception details:", exc_info=True)
        return None

def process_markdown_file(file_path, output_dir, model, tokenizer):
    """Process a single markdown file to generate a concept diagram"""
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Read the markdown file
        title, content = read_markdown_file(file_path)
        if not content:
            logger.warning(f"No usable content found in {file_path}")
            return 0
        
        # Generate diagram for the entire content
        diagram = generate_concept_diagram(model, tokenizer, title, content)
        if diagram:
            # Create base filename for diagram
            base_filename = Path(file_path).stem
            diagram_filename = f"{base_filename}_concept_diagram.md"
            diagram_path = os.path.join(output_dir, diagram_filename)
            
            with open(diagram_path, 'w', encoding='utf-8') as f:
                f.write(f"# Key Concepts: {title}\n\n")
                f.write(diagram)
                
            logger.info(f"Created concept diagram: {diagram_path}")
            return 1
        else:
            logger.warning(f"Could not generate valid diagram for {file_path}")
            return 0
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        logger.error(f"Exception details: {str(e)}", exc_info=True)
        return 0

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Generate concept diagrams from markdown files")
        parser.add_argument("--input_dir", default="chapter_markdowns", help="Directory containing markdown chapter files")
        parser.add_argument("--output_dir", default="chapter_diagrams", help="Directory to save generated diagrams")
        parser.add_argument("--model", default="google/gemma-3-27b-it", help="Hugging Face model to use")
        parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                          help="Set the logging level")
        args = parser.parse_args()
        
        # Update log level if specified
        if args.log_level:
            logger.setLevel(getattr(logging, args.log_level))
            logger.info(f"Log level set to: {args.log_level}")
        
        # Log startup information
        logger.info("Starting concept diagram generation process")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Using model: {args.model}")
        
        # Load environment variables
        load_dotenv()
        logger.debug("Environment variables loaded")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {args.output_dir}")
        
        # Check GPU availability and memory
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU memory available: {gpu_memory:.2f} GB")
            if gpu_memory < 10:  # Less than 10GB
                logger.warning("Low GPU memory detected. This might affect performance.")
        
        # Load the model
        try:
            logger.info(f"Loading model: {args.model}")
            model, tokenizer = load_model(args.model, device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error("Exception details:", exc_info=True)
            return
        
        # Find all markdown files in the input directory
        try:
            markdown_files = [f for f in os.listdir(args.input_dir) if f.endswith('.md')]
        except Exception as e:
            logger.error(f"Error accessing input directory {args.input_dir}: {str(e)}")
            return
        
        if not markdown_files:
            logger.warning(f"No markdown files found in {args.input_dir}")
            return
        
        logger.info(f"Found {len(markdown_files)} markdown files to process")
        
        # Process each markdown file
        total_diagrams = 0
        successful_files = 0
        failed_files = 0
        
        for file_name in tqdm(markdown_files, desc="Generating diagrams"):
            try:
                file_path = os.path.join(args.input_dir, file_name)
                logger.debug(f"Processing file: {file_path}")
                
                diagrams_created = process_markdown_file(file_path, args.output_dir, model, tokenizer)
                total_diagrams += diagrams_created
                
                if diagrams_created > 0:
                    successful_files += 1
                else:
                    failed_files += 1
                    logger.warning(f"No diagrams created for {file_name}")
                
            except Exception as e:
                failed_files += 1
                logger.error(f"Error processing {file_name}: {str(e)}")
                logger.debug("Exception details:", exc_info=True)
                continue
        
        # Log final statistics
        logger.info("\nProcessing completed!")
        logger.info(f"Total files processed: {len(markdown_files)}")
        logger.info(f"Successful files: {successful_files}")
        logger.info(f"Failed files: {failed_files}")
        logger.info(f"Total diagrams created: {total_diagrams}")
        
        # Log success rate
        success_rate = (successful_files / len(markdown_files)) * 100
        if success_rate < 50:
            logger.warning(f"Low success rate: {success_rate:.1f}%")
        else:
            logger.info(f"Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        logger.critical(f"Critical error in main process: {str(e)}")
        logger.critical("Exception details:", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("Fatal error in main program", exc_info=True)
        sys.exit(1)




