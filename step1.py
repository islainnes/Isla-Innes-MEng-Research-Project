import time
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
from autogen import config_list_from_json, AssistantAgent, UserProxyAgent, Cache
from functools import lru_cache
import gc
from database_extract import load_faiss_index, query_similar_papers, get_paper_context
import tempfile
from huggingface_hub import login
from dotenv import load_dotenv

# Clear CUDA cache and set PyTorch memory management
torch.cuda.empty_cache()
torch.backends.cuda.max_memory_split_size = 1024 * 1024 * 1024  # 1GB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Set up model configuration
os.environ["OAI_CONFIG_LIST"] = json.dumps([
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_client_cls": "CustomLlama2Client",
        "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
        "n": 1,
        "params": {
            "max_new_tokens": 1000,
            "do_sample": False,  # Changed to False for greedy decoding
            "num_beams": 1,      # Use beam search with 1 beam (greedy)
            "pad_token_id": 2,   # Explicitly set pad token
            "eos_token_id": 2,   # Explicitly set eos token
        },
    }
])

# Load environment variables
load_dotenv()
hf_token = os.environ.get('HUGGINGFACE_TOKEN')
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
login(token=hf_token)

# Create a temporary directory for cache
cache_dir = os.path.join(tempfile.gettempdir(), 'autogen_cache')
os.makedirs(cache_dir, exist_ok=True)

# Configure cache
config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model_client_cls": ["CustomLlama2Client"]}
)


@lru_cache(maxsize=1)
def load_shared_model(model_name, device):
    """Load model once and cache it"""
    print(f"Loading model {model_name} to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ),
        max_memory={0: "70GiB", "cpu": "32GiB"},  # Increased for A100 80GB
        offload_folder="offload",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class CustomLlama2Client:
    def __init__(self, config, **kwargs):
        self.config = config
        self.model_name = config["model"]
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.gen_params = config.get("params", {})
        self.model, self.tokenizer = load_shared_model(
            self.model_name, self.device)

    def _format_chat_prompt(self, messages):
        formatted_prompt = "<s>[INST] "
        system_message = next((m["content"]
                              for m in messages if m["role"] == "system"), None)
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
            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True)
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


def generate_report(topic, max_retries=3, num_papers=3):
    """Generate a report with a limited number of retries"""
    # Get paper context and limit to 3 papers
    full_context, all_relevant_papers = get_paper_context(topic)
    # Take only the first 3 papers
    relevant_papers = all_relevant_papers[:3]

    # Store metadata about papers including similarity scores
    paper_metadata = [
        {
            "title": paper.get('title', 'Untitled'),
            "authors": paper.get('authors', []),
            "year": paper.get('year', 'n.d.'),
            "abstract": paper.get('abstract', ''),
            "similarity": paper.get('similarity', 0),
            "content": paper.get('content', '') or paper.get('excerpt', ''),
            "id": paper.get('id', paper.get('filename', 'unknown'))
        }
        for paper in relevant_papers
    ]

    # Reconstruct context with only 3 papers
    context = "\n\n".join([
        f"Paper: {paper['title']}\n" +
        f"Abstract: {paper['abstract']}\n" +
        f"Content: {paper['content']}"
        for paper in paper_metadata
    ])

    # Create a formatted reference list from the actual paper data
    reference_list = "\n".join([
        f"[{i+1}] {paper['title']} ({paper['year']})"
        for i, paper in enumerate(paper_metadata)
    ])

    prompt = f"""Write a comprehensive academic report and literature review on: {topic}

{context}

IMPORTANT: You must ONLY use and reference the papers provided above. Do not introduce or reference any other papers or sources.

IMPORTANT GUIDELINES:
1. Total report length must be approximately 1000 words.
2. Section lengths should be:
   - INTRODUCTION: ~150 words
   - METHODOLOGY: ~150 words
   - RESULTS: ~400 words
   - DISCUSSION: ~200 words
   - CONCLUSION: ~100 words
   - REFERENCES: Use EXACTLY the papers listed above, no additional references

3. CRITICAL: Only use information from the provided paper excerpts above.
4. Do not make up or reference any additional papers or research.
5. Cite papers using [1], [2], etc. based on the order in the reference list.
6. If the provided papers don't cover certain aspects, acknowledge the limitations rather than making assumptions.
7. Focus on analyzing and synthesizing the specific findings from the provided papers.
8. The REFERENCES section must exactly match the papers listed above, with no additions or modifications.

Write the report content starting with ## INTRODUCTION and ending with ## REFERENCES."""

    writer_system_message = """You are a technical writer. Write clear, structured academic reports using markdown headers.
    Always include the following sections with ## headers:
    ## INTRODUCTION
    ## METHODOLOGY
    ## RESULTS
    ## DISCUSSION
    ## CONCLUSION
    ## REFERENCES"""

    writer = AssistantAgent(
        name="Writer",
        system_message=writer_system_message,
        llm_config={
            "config_list": config_list,
            "cache_seed": None,  # Disable caching
            "timeout": 600,
            "max_retries": 3,
            "temperature": 0.1,
        }
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1
    )

    writer.register_model_client(model_client_cls=CustomLlama2Client)

    attempts = 0
    while attempts < max_retries:
        try:
            chat_response = user_proxy.initiate_chat(
                writer, message=prompt, silent=True)
            report = user_proxy.last_message()['content']

            # Clean the response
            cleaned_report = clean_response(report)

            # Check if we have a complete report
            if "## CONCLUSION" in cleaned_report and not cleaned_report.strip().endswith("## CONCLUSION"):
                return cleaned_report, paper_metadata

            print(
                f"\nAttempt {attempts + 1} produced incomplete response. Retrying...")
            attempts += 1

        except Exception as e:
            print(f"\nError during attempt {attempts + 1}: {str(e)}")
            attempts += 1

    # If we've exhausted retries, return the best response we have
    print("\nWarning: Could not generate complete report after maximum retries.")
    return cleaned_report.strip(), paper_metadata


def generate_research_questions(domain):
    """Generate research questions within a specific domain"""
    prompt = f"""You are tasked with generating research questions in {domain}.

Generate 20 specific research questions about current challenges and emerging technologies in {domain}.

Format your response EXACTLY like this example (but with {domain} questions):
1. How do quantum tunneling effects impact the performance of next-generation semiconductor devices?
2. What are the limitations of current silicon-based transistors at sub-5nm scales?
3. How can machine learning optimize power consumption in IoT devices?

Generate exactly 20 questions, numbered 1-20. Make each question specific and suitable for a literature review."""

    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1
    )

    question_generator = AssistantAgent(
        name="QuestionGenerator",
        system_message="Generate numbered research questions only. No additional text.",
        llm_config={
            "config_list": config_list,
            "cache_seed": None,  # Disable caching
            "temperature": 0.1,
        }
    )

    question_generator.register_model_client(
        model_client_cls=CustomLlama2Client)

    chat_response = user_proxy.initiate_chat(
        question_generator, message=prompt, silent=True)
    response_text = user_proxy.last_message()['content'].strip()

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
        print(
            f"Warning: Only generated {len(questions)} questions instead of 20")
    elif len(questions) > 20:
        questions = questions[:20]

    if not questions:
        # Fallback questions if extraction fails
        questions = [
            f"What are the latest advances in {domain}?",
            f"How can artificial intelligence improve {domain}?",
            # ... add more fallback questions ...
        ]
        print("Warning: Using fallback questions due to parsing error")

    return questions


def main():
    domain = "Electronic Engineering"
    print(f"\nGenerating research questions in: {domain}")

    # Create output directory
    output_dir = "litreviews"
    os.makedirs(output_dir, exist_ok=True)

    # Generate research questions
    research_questions = generate_research_questions(domain)

    # Generate review for each question
    for i, question in enumerate(research_questions, 1):
        print(f"\nGenerating review {i}/20: {question}")

        try:
            # Now getting both report and metadata
            report, paper_metadata = generate_report(question)
            cleaned_report = clean_response(report)

            # Convert report to JSON structure with complete paper metadata
            report_data = {
                "question": question,
                "domain": domain,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sections": extract_sections(cleaned_report),
                # Now includes full paper details with similarity
                "referenced_papers": paper_metadata,
                "metadata": {
                    "total_papers": len(paper_metadata),
                    "average_similarity": sum(p['similarity'] for p in paper_metadata) / len(paper_metadata) if paper_metadata else 0,
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }

            # Save to JSON file
            output_file = os.path.join(
                output_dir, f"review_{i:02d}_{int(time.time())}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)

            # Clear some memory
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error generating review {i}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
