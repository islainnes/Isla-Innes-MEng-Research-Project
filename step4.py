import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os
from dotenv import load_dotenv  # Add this import

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
    """Load the original review and recommendations data"""
    # Load original review
    with open('litreviews/review_01_1739874785.json', 'r') as f:
        review_data = json.load(f)
    
    # Load rewrite actions from final_points.json
    with open('output/final_points.json', 'r') as f:
        rewrite_data = json.load(f)  # Load all sections' actions
    
    return review_data, rewrite_data

def generate_rewrite_prompt(section_name: str, original_text: str, improvement_points: list) -> str:
    """Generate a prompt for rewriting any section based on improvement points"""
    formatted_points = "\n".join(f"   {point}" for point in improvement_points)
    
    prompt = f"""<s>[INST] Rewrite the following {section_name} section while maintaining its original structure. Address these specific improvement points:

{formatted_points}

Original text:
{original_text}

Please provide a rewrite that:
1. Maintains the same basic structure and technical depth
2. Addresses the improvement points listed above
3. Uses clear, accessible language while preserving technical accuracy
4. Ensures smooth transitions between ideas
5. Maintains proper citations and technical terminology

[/INST]"""
    
    return prompt

def rewrite_section(model, tokenizer, section_name: str, original_text: str, improvement_points: list) -> str:
    """Rewrite any section using the LLM based on improvement points"""
    # Existing rewrite_introduction function logic, but generalized for any section
    prompt = generate_rewrite_prompt(section_name, original_text, improvement_points)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,     # Longer context
                temperature=0.2,        # More focused
                top_k=20,              # More conservative
                do_sample=False,       # Deterministic
                num_beams=4,          # More beam search
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split('[/INST]')[-1].strip()
        
        # Enhanced cleanup
        response = response.replace('\n\n\n', '\n\n')
        response = response.replace('  ', ' ')
        response = response.replace('**', '').replace('*', '').replace('#', '')
        
        # Remove section headers and titles
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if not any(line.startswith(prefix) for prefix in ['Title:', 'Introduction:', 'Definition and Background:', '###', '####', '---']):
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        
        # Ensure proper paragraph structure
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) > 3:
            paragraphs = [paragraphs[0], '\n\n'.join(paragraphs[1:-1]), paragraphs[-1]]
        
        response = '\n\n'.join(paragraphs)
        
        # Clean up special characters and encoding issues
        response = ''.join(char for char in response if ord(char) < 65536)
        
        # Validate response
        if not response or len(response.split()) < 50:  # Basic validation
            raise ValueError("Generated response is too short or empty")
            
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

def main():
    # Load the model
    model, tokenizer = load_model()
    
    # Load review data and recommendations
    review_data, rewrite_data = load_review_data()
    
    # Create output directory for rewritten reports
    rewritten_dir = Path("rewritten_reports")
    rewritten_dir.mkdir(exist_ok=True)
    
    # Initialize the improved report structure
    improved_report = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "original_file": "litreviews/review_01_1739874785.json",
            "improvement_points": rewrite_data
        },
        "sections": {}
    }
    
    # List of sections to process (excluding References)
    sections_to_process = [
        section for section in review_data['sections'].keys() 
        if section != 'REFERENCES'
    ]
    
    # Process each section
    for section_name in sections_to_process:
        print(f"\nRewriting {section_name}...")
        
        original_text = review_data['sections'][section_name]
        improvement_points = rewrite_data.get(section_name, [])
        
        if improvement_points:  # Only rewrite if we have improvement points
            improved_text = rewrite_section(
                model,
                tokenizer,
                section_name,
                original_text,
                improvement_points
            )
        else:
            improved_text = original_text
            print(f"No improvement points found for {section_name}, keeping original")
        
        # Add to improved report
        improved_report['sections'][section_name] = {
            "original": original_text,
            "improved": improved_text
        }
    
    # Save the improved report
    output_path = rewritten_dir / f"improved_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(improved_report, f, indent=2, ensure_ascii=False)
    
    print(f"\nImproved report saved to {output_path}")

if __name__ == "__main__":
    main()

