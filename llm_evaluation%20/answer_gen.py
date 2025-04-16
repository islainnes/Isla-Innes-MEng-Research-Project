from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import os
import time
import gc
from step1 import generate_research_questions

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
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    # Load questions from gs_answers.json
    with open('gs_answers.json', 'r', encoding='utf-8') as f:
        gs_data = json.load(f)
    research_questions = [qa['question'] for qa in gs_data['qa_pairs']]
    
    all_results = []
    
    # Process each model separately - only original and step5 model
    model_configs = [
        ("original", "mistralai/Mistral-7B-Instruct-v0.2"),
        ("improved_only", "./fine_tuned_model_improved_only")
    ]
    
    for model_name, model_path in model_configs:
        print(f"\nLoading {model_name} model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            offload_folder="offload_folder"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Process all questions with current model
        for i, question in enumerate(research_questions, 1):
            print(f"Processing question {i}/20 with {model_name} model")
            response = generate_response(question, model, tokenizer)
            
            # Find or create result dictionary for this question
            if i > len(all_results):
                all_results.append({
                    "question_number": i,
                    "question": question,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            all_results[i-1][f"{model_name}_model_response"] = response
        
        # Clear model from GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results
    output_dir = "model_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"model_responses_{int(time.time())}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
