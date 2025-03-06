import time
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from autogen import config_list_from_json, AssistantAgent, UserProxyAgent
from functools import lru_cache
import gc
from database_extract import get_paper_context

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Clear CUDA cache and set PyTorch memory management
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Model configuration
MODEL_CONFIG = {
    "model": "mistralai/Mistral-Small-24B-Instruct-2501",
    "model_client_cls": "CustomLlama2Client",
    "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
    "n": 1,
    "params": {
        "max_new_tokens": 1000,
        "top_k": 50,
        "temperature": 0.1,
        "do_sample": True,
    },
    "memory_config": {
        "gpu": "70GiB",
        "cpu": "32GiB"
    }
}

os.environ["OAI_CONFIG_LIST"] = json.dumps([MODEL_CONFIG])

# HuggingFace login
from huggingface_hub import login
login(token=os.getenv('HUGGINGFACE_TOKEN'))

@lru_cache(maxsize=1)
def load_shared_model(model_name, device):
    """Load model once and cache it"""
    print(f"Loading model {model_name} to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: MODEL_CONFIG["memory_config"]["gpu"], 
                   "cpu": MODEL_CONFIG["memory_config"]["cpu"]},
        offload_folder="offload",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

class CustomLlama2Client:
    def __init__(self, config, **kwargs):
        self.config = config
        self.model_name = config["model"]
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.gen_params = config.get("params", {})
        self.model, self.tokenizer = load_shared_model(self.model_name, self.device)

    def _format_chat_prompt(self, messages):
        formatted_prompt = "<s>[INST] "
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
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

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.gen_params
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    
    for line in text.split('\n'):
        if line.startswith('## '):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line.replace('## ', '').strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Add the last section
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

def validate_report_sections(report):
    """Validate that all required sections are present in the report"""
    required_sections = [
        "## INTRODUCTION",
        "## METHODOLOGY",
        "## RESULTS",
        "## DISCUSSION",
        "## FUTURE RESEARCH",
        "## CONCLUSION",
        "## REFERENCES"
    ]
    return all(section in report for section in required_sections)

def generate_research_questions(domain):
    """Generate research questions within a specific domain using direct LLM calls"""
    prompt = f"""You are tasked with generating research questions in {domain}.

Generate 20 specific research questions about current challenges and emerging technologies in {domain}.

Format your response EXACTLY like this example (but with {domain} questions):
1. How do quantum tunneling effects impact the performance of next-generation semiconductor devices?
2. What are the limitations of current silicon-based transistors at sub-5nm scales?
3. How can machine learning optimize power consumption in IoT devices?

Generate exactly 20 questions, numbered 1-20. Make each question specific and suitable for a literature review."""

    llm_client = CustomLlama2Client(MODEL_CONFIG)
    
    response = llm_client.create({
        "messages": [
            {"role": "system", "content": "Generate numbered research questions only. No additional text."},
            {"role": "user", "content": prompt}
        ]
    })
    
    response_text = llm_client.message_retrieval(response)[0].strip()
    
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
        print(f"Warning: Only generated {len(questions)} questions instead of 20")
    elif len(questions) > 20:
        questions = questions[:20]
    
    return questions

def generate_report(topic, max_retries=3, num_papers=3):
    """Generate a report with a limited number of retries using direct LLM calls"""
    context, relevant_papers = get_paper_context(topic, num_papers)
    
    reference_list = "\n".join([
        f"[{i+1}] {paper['title']} ({paper['year']})"
        for i, paper in enumerate(relevant_papers)
    ])
    
    prompt = f"""Write a comprehensive academic report and literature review on: {topic}

{context}

References to use:
{reference_list}

IMPORTANT: You must ONLY use and reference the papers provided above. Do not introduce or reference any other papers or sources.

IMPORTANT GUIDELINES:
1. Total report length must be approximately 1200 words.
2. Section lengths should be:
   - INTRODUCTION: ~150 words
   - METHODOLOGY: ~150 words
   - RESULTS: ~350 words
   - DISCUSSION: ~200 words
   - FUTURE RESEARCH: ~200 words (Identify gaps in current research and suggest promising future research directions)
   - CONCLUSION: ~150 words
   - REFERENCES: Use EXACTLY the papers listed above, no additional references

3. CRITICAL: Only use information from the provided paper excerpts above.
4. Do not make up or reference any additional papers or research.
5. Cite papers using [1], [2], etc. based on the order in the reference list.
6. If the provided papers don't cover certain aspects, acknowledge the limitations rather than making assumptions.
7. Focus on analyzing and synthesizing the specific findings from the provided papers.
8. In the FUTURE RESEARCH section:
   - Identify clear gaps in the current research
   - Suggest specific research questions that could address these gaps
   - Propose potential methodological approaches for future studies
   - Highlight emerging trends or technologies that could influence future research
9. The REFERENCES section must exactly match the papers listed above, with no additions or modifications.

Write the report content starting with ## INTRODUCTION and ending with ## REFERENCES."""

    llm_client = CustomLlama2Client(MODEL_CONFIG)
    
    attempts = 0
    while attempts < max_retries:
        try:
            response = llm_client.create({
                "messages": [
                    {"role": "system", "content": "You are a technical writer. Write clear, structured academic reports using markdown headers."},
                    {"role": "user", "content": prompt}
                ]
            })
            
            report = llm_client.message_retrieval(response)[0]
            cleaned_report = clean_response(report)
            
            if validate_report_sections(cleaned_report):
                return cleaned_report
            
            print(f"\nAttempt {attempts + 1} produced incomplete response. Retrying...")
            attempts += 1
            
        except Exception as e:
            print(f"\nError during attempt {attempts + 1}: {str(e)}")
            attempts += 1
    
    print("\nWarning: Could not generate complete report after maximum retries.")
    return cleaned_report.strip()

def main():
    domain = "Electronic Engineering"
    print(f"\nGenerating research questions in: {domain}")
    
    output_dir = "litreviews"
    os.makedirs(output_dir, exist_ok=True)
    
    research_questions = generate_research_questions(domain)
    
    for i, question in enumerate(research_questions, 1):
        print(f"\nGenerating review {i}/20: {question}")
        
        try:
            report = generate_report(question)
            cleaned_report = clean_response(report)
            
            _, relevant_papers = get_paper_context(question)
            
            report_data = {
                "question": question,
                "domain": domain,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sections": extract_sections(cleaned_report),
                "referenced_papers": [
                    {
                        "title": paper["title"],
                        "year": paper.get("year", "N/A"),
                        "similarity": paper.get("similarity", 0)
                    }
                    for paper in relevant_papers
                ]
            }
            
            output_file = os.path.join(output_dir, f"review_{i:02d}_{int(time.time())}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error generating review {i}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
