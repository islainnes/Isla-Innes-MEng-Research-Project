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
import logging
import sys
import time
from pathlib import Path
import argparse

def setup_logging(log_level=logging.INFO):
    """Configure logging with both file and console output."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fine_tuning_{timestamp}.log"
    
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

# Load environment variables
load_dotenv()
logger.debug("Environment variables loaded")

# Add HuggingFace login for accessing gated models
try:
    login(token=os.getenv('HUGGINGFACE_TOKEN'))
    logger.info("Successfully logged in to HuggingFace")
except Exception as e:
    logger.error(f"Failed to log in to HuggingFace: {str(e)}")
    raise

def load_text_files(directory):
    """Load text content from markdown files."""
    texts = []
    logger.info(f"Looking for markdown files in directory: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist")
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    files = [f for f in os.listdir(directory) if f.endswith('.md')]
    logger.info(f"Found {len(files)} markdown files")
    
    for file in files:
        file_path = os.path.join(directory, file)
        logger.debug(f"Processing file: {file}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(content)
                logger.debug(f"Successfully loaded content from {file}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file}: {e}")
            logger.error("Exception details:", exc_info=True)
    
    logger.info(f"Total number of texts extracted: {len(texts)}")
    if len(texts) == 0:
        logger.error("No texts were extracted from the markdown files")
        raise ValueError("No texts were extracted from the markdown files")
    
    return texts

def prepare_dataset(texts, tokenizer):
    """Tokenize texts and prepare them for training."""
    logger.info("Preparing dataset from texts")
    try:
        logger.debug(f"Tokenizing {len(texts)} texts")
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        dataset = Dataset.from_dict(encodings)
        logger.info(f"Successfully created dataset with {len(dataset)} examples")
        return dataset
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        logger.error("Exception details:", exc_info=True)
        raise

def load_json_files(directory):
    """Load improved content from JSON files in the specified directory."""
    texts = []
    logger.info(f"Looking for JSON files in directory: {directory}")
    
    if not os.path.exists(directory):
        logger.error(f"Directory '{directory}' does not exist")
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    logger.info(f"Found {len(files)} JSON files")
    
    # These are the sections we want to extract, in order
    sections = ['INTRODUCTION', 'METHODOLOGY', 'RESULTS', 'DISCUSSION', 'CONCLUSION']
    
    for file in files:
        file_path = os.path.join(directory, file)
        logger.debug(f"Processing file: {file}")
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
                                logger.debug(f"Successfully extracted improved {section} from {file}")
                else:
                    logger.warning(f"No 'sections' field found in {file}")
                    logger.debug(f"Available keys in {file}: {list(data.keys())}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"Error reading JSON from {file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file}: {e}")
            logger.error("Exception details:", exc_info=True)
    
    logger.info(f"Total number of texts extracted: {len(texts)}")
    if len(texts) == 0:
        logger.error("No texts were extracted from the JSON files")
        raise ValueError("No texts were extracted from the JSON files")
    
    return texts

def train_model(model, dataset, output_dir, num_train_epochs=3, per_device_train_batch_size=4):
    """Train the model on the dataset."""
    logger.info("Starting model training")
    logger.info(f"Training parameters: epochs={num_train_epochs}, batch_size={per_device_train_batch_size}")
    
    try:
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=100,
            save_total_limit=2,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
        )
        logger.debug("Training arguments configured")

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        logger.debug("Trainer initialized")

        # Train the model
        logger.info("Beginning training process")
        trainer.train()
        logger.info("Training completed successfully")

        # Save the model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        logger.info("Model saved successfully")

        return trainer
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error("Exception details:", exc_info=True)
        raise

def main():
    """Main function to run the fine-tuning process."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Fine-tune a language model on academic text.')
        parser.add_argument('--input_dir', required=True, help='Directory containing input files')
        parser.add_argument('--output_dir', required=True, help='Directory to save the fine-tuned model')
        parser.add_argument('--model_name', default='gpt2', help='Name of the base model to fine-tune')
        parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
        parser.add_argument('--log_level', default='INFO', 
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                          help='Set the logging level')
        args = parser.parse_args()

        # Setup logging
        setup_logging(getattr(logging, args.log_level))
        logger.info("Starting fine-tuning process")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Base model: {args.model_name}")

        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            logger.debug(f"Created output directory: {args.output_dir}")

        # Load the tokenizer and model
        logger.info(f"Loading tokenizer and model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        logger.info("Model and tokenizer loaded successfully")

        # Load and prepare the dataset
        logger.info("Loading dataset from input files")
        texts = load_json_files(args.input_dir)  # or load_text_files depending on your data
        dataset = prepare_dataset(texts, tokenizer)
        logger.info(f"Dataset prepared with {len(dataset)} examples")

        # Train the model
        trainer = train_model(
            model, 
            dataset, 
            args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size
        )

        logger.info("Fine-tuning process completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        logger.error("Exception details:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
