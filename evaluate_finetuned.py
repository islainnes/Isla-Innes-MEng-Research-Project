from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import os
import time
import gc
from step1 import generate_research_questions
from peft import PeftModel

def generate_response(prompt, model, tokenizer, max_length=512):
    """Generate a response from the model given a prompt."""
    # Format the prompt according to Mistral's instruction format
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from the response
    response = response.replace(formatted_prompt, "").strip()
    return response

def main():
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
    )
    
    # Generate research questions
    domain = "Electronic Engineering"
    research_questions = generate_research_questions(domain)
    
    # Setup output
    output_dir = "model_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"model_responses.json")
    
    # Load existing results
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    total_questions = len(research_questions)
    start_time = time.time()

    # Load fine-tuned model
    model_path = "./fine_tuned_model_improved_only"
    print(f"\nLoading fine-tuned model... {time.strftime('%H:%M:%S')}")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            offload_folder="offload_folder"
        )
        
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Process questions
        for i, question in enumerate(research_questions, 1):
            question_start = time.time()
            print(f"Processing question {i}/{total_questions} with fine-tuned model")
            
            try:
                response = generate_response(question, model, tokenizer)
                question_time = time.time() - question_start
                print(f"Question {i} took {question_time:.2f} seconds")
                
                # Ensure the dictionary exists for this question
                if i > len(all_results):
                    all_results.append({
                        "question_number": i,
                        "question": question,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                # Add the response with the specific model name
                all_results[i-1]["improved_only_response"] = response
                
                # Save after each response
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2)
                
            except Exception as e:
                print(f"Error processing question {i}: {str(e)}")
                if i <= len(all_results):
                    all_results[i-1]["improved_only_response"] = f"Error: {str(e)}"
                continue
            
    finally:
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main() 