from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
import json
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Add HuggingFace login for accessing gated models
login(token=os.getenv('HUGGINGFACE_TOKEN'))

def load_text_files(directory):
    """Load text content from markdown files."""
    texts = []
    print(f"Looking for markdown files in directory: {directory}")
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    files = [f for f in os.listdir(directory) if f.endswith('.md')]
    print(f"Found {len(files)} markdown files")
    
    for file in files:
        file_path = os.path.join(directory, file)
        print(f"Processing file: {file}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(content)
                print(f"Successfully loaded content from {file}")
        except Exception as e:
            print(f"Unexpected error processing {file}: {e}")
    
    print(f"Total number of texts extracted: {len(texts)}")
    if len(texts) == 0:
        raise ValueError("No texts were extracted from the markdown files")
    
    return texts

def prepare_dataset(texts, tokenizer):
    """Tokenize texts and prepare them for training."""
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return Dataset.from_dict(encodings)

def load_json_files(directory):
    """Load improved content from JSON files in the specified directory."""
    texts = []
    print(f"Looking for JSON files in directory: {directory}")
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files")
    
    # These are the sections we want to extract, in order
    sections = ['INTRODUCTION', 'METHODOLOGY', 'RESULTS', 'DISCUSSION', 'CONCLUSION']
    
    for file in files:
        file_path = os.path.join(directory, file)
        print(f"Processing file: {file}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if 'sections' key exists
                if 'sections' in data:
                    # Create a prompt template for each section's improved content
                    for section in sections:
                        if section in data['sections']:
                            section_data = data['sections'][section]
                            # Extract only the 'improved' content if available
                            if isinstance(section_data, dict) and 'improved' in section_data:
                                improved_text = section_data['improved']
                                # Create a formatted text with section name and content
                                formatted_text = f"### {section}:\n{improved_text}\n\n"
                                texts.append(formatted_text)
                                print(f"Successfully extracted improved {section} from {file}")
                else:
                    print(f"Warning: No 'sections' field found in {file}")
                    print(f"Available keys in {file}: {list(data.keys())}")
                    
        except json.JSONDecodeError as e:
            print(f"Error reading JSON from {file}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {file}: {e}")
    
    print(f"Total number of texts extracted: {len(texts)}")
    if len(texts) == 0:
        raise ValueError("No texts were extracted from the JSON files")
    
    return texts

def main():
    # Define model and tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Define directory for markdown files
    directory = 'chapter_markdowns'
    
    print(f"\nStarting fine-tuning using content from markdown files in {directory}")
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        offload_folder="offload",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load texts from markdown files
    texts = load_text_files(directory)
    
    # Prepare dataset
    dataset = prepare_dataset(texts, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_dir='./logs',
        fp16=True,
        gradient_checkpointing=True,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    print("Completed fine-tuning using markdown content")

if __name__ == "__main__":
    main()

