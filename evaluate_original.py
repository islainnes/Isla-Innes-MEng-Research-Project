from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import os
import time
import gc
from step1 import generate_research_questions

# Keep your existing generate_response function
def generate_response(prompt, model, tokenizer, max_length=512):
    # ... existing generate_response code ...

def main():
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Generate research questions
    domain = "Electronic Engineering"
    research_questions = generate_research_questions(domain)
    
    # Setup output
    output_dir = "model_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"model_responses.json")
    
    # Load or create results file
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    total_questions = len(research_questions)
    start_time = time.time()

    # Load original model
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"\nLoading original model... {time.strftime('%H:%M:%S')}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Process questions
        for i, question in enumerate(research_questions, 1):
            # ... existing question processing code ...
            
    finally:
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main() 