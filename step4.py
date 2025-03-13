import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os
from dotenv import load_dotenv  # Add this import
import re

# Load environment variables
load_dotenv()

# Add HuggingFace login for accessing gated models
login(token=os.getenv('HUGGINGFACE_TOKEN'))

# Add at the start of the script
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_model():
    """Load the Mistral model with improved configuration for better text generation"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name} to {device}...")
    
    # Use 8-bit quantization instead of 4-bit for better quality
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            offload_folder="offload",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_compile=True  # Optional: might improve memory usage
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Set generation parameters for better quality
        model.generation_config.max_new_tokens = 500
        model.generation_config.temperature = 0.7
        model.generation_config.top_k = 50
        model.generation_config.do_sample = True
        model.generation_config.num_beams = 4
        model.generation_config.no_repeat_ngram_size = 3
        model.generation_config.early_stopping = True
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Falling back to 4-bit quantization...")
        
        # Fallback to 4-bit if 8-bit fails
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            offload_folder="offload",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

def load_review_data():
    """Load all review data and their corresponding recommendations"""
    review_data_list = []
    
    # Get only review_01 files directly in litreviews folder
    litreviews_path = Path('litreviews')
    review_files = list(litreviews_path.glob('review_01*.json'))
    
    # Sort files to process in order
    review_files.sort()
    
    for review_path in review_files:
        print(f"Checking file: {review_path}")
        
        # Get the base name without extension for finding the corresponding output folder
        review_base = review_path.stem  # e.g., 'review_01'
        
        # Find corresponding final_points.json in output directory
        final_points_path = Path('output') / 'review_01' / 'final_points.json'
        
        if not final_points_path.exists():
            print(f"No final_points.json found for review_01")
            continue
            
        try:
            # Load review data
            with open(review_path, 'r') as f:
                review_data = json.load(f)
                
            # Load rewrite points
            with open(final_points_path, 'r') as f:
                rewrite_data = json.load(f)
                
            review_data_list.append({
                'review_path': review_path,
                'review_data': review_data,
                'rewrite_data': rewrite_data
            })
            print(f"Successfully loaded data for review_01")
            
        except Exception as e:
            print(f"Error loading data for review_01: {str(e)}")
            continue
    
    if not review_data_list:
        raise FileNotFoundError("No valid review data found")
        
    return review_data_list

def generate_rewrite_prompt(section_name: str, original_text: str, improvement_points: list) -> str:
    """Generate a more specific and directive prompt"""
    section_purpose = get_section_purpose(section_name)
    formatted_points = "\n".join(f"- {point}" for point in improvement_points)
    
    prompt = f"""As a technical writing expert, rewrite this {section_name} section. This section's primary purpose is to {section_purpose}.

ORIGINAL TEXT:
{original_text}

REQUIRED IMPROVEMENTS:
{formatted_points}

IMPORTANT REQUIREMENTS:
1. The rewritten text MUST be longer and more detailed than the original
2. MUST address EVERY improvement point listed above
3. MUST maintain all original citations [X] and technical terminology
4. MUST maintain proper academic/technical writing style
5. MUST end with complete sentences and proper paragraph structure

REWRITTEN TEXT:
"""
    return prompt

def rewrite_section(model, tokenizer, section_name: str, original_text: str, improvement_points: list) -> str:
    """Rewrite any section using the LLM based on improvement points"""
    prompt = generate_rewrite_prompt(section_name, original_text, improvement_points)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                top_k=50,
                do_sample=True,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add debugging
        print(f"\nRaw response for {section_name}:")
        print(response)
        
        # Extract the rewritten text
        response = response.split("REWRITTEN TEXT:")[-1].strip()
        
        print(f"\nExtracted response for {section_name}:")
        print(response)
        
        # Clean up the response
        response = clean_response(response)
        
        return response
            
    except Exception as e:
        print(f"Error during text generation for {section_name}: {str(e)}")
        return original_text

def validate_technical_terms(original_text, new_text, terms_to_check):
    """Ensure technical terms remain consistent and properly formatted"""
    for term in terms_to_check:
        if term in original_text and term not in new_text:
            return False
    return True

def get_section_purpose(section_name: str) -> str:
    """Return the primary purpose of each section"""
    purposes = {
        "INTRODUCTION": "Provide context, state the problem, and outline the report's structure",
        "METHODOLOGY": "Detail the research methods, procedures, and analytical approaches used",
        "RESULTS": "Present the key findings and data without interpretation",
        "DISCUSSION": "Interpret the results, compare with literature, and discuss implications",
        "CONCLUSION": "Summarize key findings and their significance"
    }
    return purposes.get(section_name, "Present information clearly and accurately")

def validate_rewritten_section(original: str, rewritten: str, improvement_points: list) -> bool:
    """Validate that the rewritten section meets quality criteria"""
    
    # Relax the length check - should be at least 80% of original length
    if len(rewritten.split()) < len(original.split()) * 0.8:
        print("Failed length validation")
        return False
        
    # Check for truncation (ending mid-sentence)
    if rewritten.rstrip()[-1] not in ".!?":
        print("Failed sentence ending validation")
        return False
        
    # Ensure at least 50% of original citations are preserved
    original_citations = re.findall(r'\[\d+\]', original)
    rewritten_citations = re.findall(r'\[\d+\]', rewritten)
    if len(original_citations) > 0 and len(rewritten_citations) < len(original_citations) * 0.5:
        print("Failed citation preservation validation")
        return False
        
    # Relax technical terms check - at least 50% should be preserved
    technical_terms = extract_technical_terms(original)
    if technical_terms:
        preserved_terms = sum(1 for term in technical_terms if term.lower() in rewritten.lower())
        if preserved_terms < len(technical_terms) * 0.5:
            print("Failed technical terms validation")
            return False
    
    return True

def extract_technical_terms(text: str) -> set:
    """Extract important technical terms from text"""
    # This is a simple implementation - could be enhanced with NLP
    words = text.split()
    technical_terms = set()
    
    # Look for capitalized terms and terms in parentheses
    for i, word in enumerate(words):
        if (word[0].isupper() and i > 0) or '(' in word or ')' in word:
            technical_terms.add(word.strip('().,'))
            
    return technical_terms

def clean_response(text: str) -> str:
    """Clean up the generated response"""
    # Remove any remaining prompt text
    text = re.sub(r'^.*?REWRITTEN TEXT:', '', text, flags=re.DOTALL)
    
    # Clean up formatting
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('  ', ' ')
    text = text.replace('**', '').replace('*', '')
    
    # Remove section headers and titles
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith(('Title:', 'Introduction:', '###', '####', '---'))]
    
    # Rejoin and ensure proper spacing
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def main():
    # Load the model
    model, tokenizer = load_model()
    
    # Load all review data and recommendations
    review_data_list = load_review_data()
    
    # Create output directory for rewritten reports
    rewritten_dir = Path("rewritten_reports")
    rewritten_dir.mkdir(exist_ok=True)
    
    # Process each review
    for review_item in review_data_list:
        try:
            review_path = review_item['review_path']
            review_data = review_item['review_data']
            rewrite_data = review_item['rewrite_data']
            
            print(f"\nProcessing review: {review_path}")
            
            improved_report = {
                "metadata": {
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                    "original_file": str(review_path),
                    "improvement_points": rewrite_data,
                    "processing_status": {}  # Track success/failure of each section
                },
                "sections": {}
            }
            
            sections_to_process = [
                section for section in review_data['sections'].keys() 
                if section != 'REFERENCES'
            ]
            
            for section_name in sections_to_process:
                try:
                    print(f"  Rewriting {section_name}...")
                    
                    original_text = review_data['sections'][section_name]
                    improvement_points = rewrite_data.get(section_name, [])
                    
                    if improvement_points:
                        improved_text = rewrite_section(
                            model,
                            tokenizer,
                            section_name,
                            original_text,
                            improvement_points
                        )
                        status = "improved"
                    else:
                        improved_text = original_text
                        status = "no_improvements_needed"
                    
                    improved_report['metadata']['processing_status'][section_name] = status
                    improved_report['sections'][section_name] = {
                        "original": original_text,
                        "improved": improved_text
                    }
                    
                except Exception as e:
                    print(f"Error processing section {section_name}: {str(e)}")
                    improved_report['metadata']['processing_status'][section_name] = "error"
                    improved_report['sections'][section_name] = {
                        "original": original_text,
                        "improved": original_text,
                        "error": str(e)
                    }
            
            # Save the improved report with status information
            output_filename = f"improved_{review_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            output_path = rewritten_dir / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(improved_report, f, indent=2, ensure_ascii=False)
            
            print(f"  Improved report saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing review {review_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()



