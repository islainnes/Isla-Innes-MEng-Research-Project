import autogen
from typing import Dict, List
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import lru_cache
from huggingface_hub import login
from pathlib import Path
import time
from types import SimpleNamespace
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Clear CUDA cache and set memory management
torch.cuda.empty_cache()
torch.backends.cuda.max_memory_split_size = 1024 * 1024 * 1024  # 1GB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# Set up the model configuration
os.environ["OAI_CONFIG_LIST"] = json.dumps(
    [
        {
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
            "timeout": 120,
            "retry_on_error": True,
            "max_retries": 3
        }
    ]
)

# Add HuggingFace login for accessing gated models
login(token=os.getenv('HUGGINGFACE_TOKEN'))

@lru_cache(maxsize=1)
def load_shared_model(model_name, device):
    """Load model once and cache it for reuse"""
    print(f"Loading model {model_name} to {device}...")
    
    try:
        # More aggressive memory optimization settings
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
            max_memory={
                0: "70GiB",  # Increased for A100 80GB
                "cpu": "16GiB"
            },
            offload_folder="offload",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Successfully loaded model {model_name}")
        return model, tokenizer
    except Exception as e:
        print(f"ERROR loading model {model_name}: {e}")
        # Re-raise to ensure the error is visible
        raise

class CustomLlama2Client:
    """Custom model client implementation for Llama-2 with AutoGen."""
    
    def __init__(self, config, **kwargs):
        """Initialize the client."""
        self.config = config
        self.model_name = config["model"]
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.gen_params = config.get("params", {})
        
        print(f"CustomLlama2Client config: {config}")
        
        # Use shared model and tokenizer
        self.model, self.tokenizer = load_shared_model(self.model_name, self.device)

    def _format_chat_prompt(self, messages):
        """Format messages into Mistral's chat format."""
        # Get the system message and user message
        system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_message = next((m["content"] for m in messages if m["role"] == "user"), None)
        
        # Extract the section name and content from the user message
        if user_message:
            # Split on "Section to review:" to get the actual content
            parts = user_message.split("Section to review:")
            if len(parts) > 1:
                section_content = parts[1].strip()
            else:
                section_content = user_message.strip()
            
            # Clean up the section content to remove any duplicate prompts
            if "Please analyze the content and provide:" in section_content:
                section_content = section_content.split("Please analyze the content and provide:")[0].strip()
            
            # Clean up any previous responses that might have been included
            if "Feedback:" in section_content:
                section_content = section_content.split("Feedback:")[0].strip()
            
            # Clean up any other potential markers
            if "Remember to:" in section_content:
                section_content = section_content.split("Remember to:")[0].strip()
        else:
            section_content = ""
        
        # Format the final prompt
        if system_message:
            return f"{system_message}\n\n{section_content}"
        else:
            return section_content

    def _create(self, params):
        """Internal method to create a response using the model."""
        # Ensure streaming is not requested
        if params.get("stream", False):
            raise NotImplementedError("Streaming not implemented for Llama 2 client.")

        num_of_responses = params.get("n", 1)
        response = SimpleNamespace()
        
        # Format the chat prompt
        prompt = self._format_chat_prompt(params["messages"])
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        response.choices = []
        response.model = self.model_name

        # Generate responses
        with torch.no_grad():
            for _ in range(num_of_responses):
                outputs = self.model.generate(
                    **inputs,
                    **self.gen_params
                )
                
                # Decode the generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the model's response
                response_text = generated_text.strip()
                
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = response_text
                choice.message.function_call = None
                response.choices.append(choice)

        return response

    def create(self, params):
        """Create a response using the model, bypassing cache."""
        return self._create(params)

    def message_retrieval(self, response):
        """Retrieve messages from the response."""
        return [choice.message.content for choice in response.choices]

    def cost(self, response) -> float:
        """Calculate the cost of the response."""
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        """Get usage statistics."""
        return {}

    def get_cache_key(self, params):
        """Override cache key generation to prevent disk caching."""
        return None  # Return None to disable caching

class DebateManager:
    """Manages the debate stage where agents discuss and resolve conflicts in their reviews."""
    
    def __init__(self):
        self.reviews = []
    
    def add_review(self, agent_name: str, review: str):
        """Add a review from an agent."""
        self.reviews.append(f"[{agent_name}]: {review}")
    
    def resolve(self) -> str:
        """Resolve conflicts and combine reviews."""
        return "\n\n".join(self.reviews)

class TechnicalAccuracyAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="technical_accuracy_agent",
            system_message="""You are a technical accuracy specialist. Your role is to:
            1. Review the technical content for accuracy
            2. Identify any technical errors or inconsistencies
            3. Suggest improvements for technical precision
            4. Ensure scientific/technical terminology is used correctly
            
            Provide ONLY feedback and suggestions. Do not rewrite or modify the content.
            Focus on identifying issues and suggesting improvements.
            
            Guidelines for your response:
            - Provide 3-5 key technical issues or suggestions
            - Be specific and actionable
            - Avoid repetition
            - Focus on technical accuracy only
            - Keep each point concise""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomLlama2Client)

    def review(self, section_name: str, section_content: str) -> str:
        """Review a section for technical accuracy."""
        chat_initiator = self.initiate_chat(
            self,
            message=f"""Please review this {section_name} section for technical accuracy.
            Provide ONLY feedback and suggestions. Do not rewrite the content.
            
            Guidelines:
            - Provide 3-5 key issues or suggestions
            - Be specific and actionable
            - Avoid repetition
            - Keep each point concise
            
            Section to review:
            {section_content}
            
            TERMINATE"""
        )
        return chat_initiator.last_message().get("content", "")

class ClarityAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="clarity_readability_agent",
            system_message="""You are a clarity and readability specialist. Your role is to:
            1. Assess the document's clarity and readability
            2. Identify unclear or confusing sections
            3. Suggest improvements for better flow and understanding
            4. Ensure appropriate language level for the target audience
            
            Provide ONLY feedback and suggestions. Do not rewrite or modify the content.
            Focus on identifying areas that need clarification.
            
            Guidelines for your response:
            - Provide 3-5 key clarity issues or suggestions
            - Be specific about which parts need improvement
            - Avoid repetition
            - Focus on readability only
            - Keep each point concise""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomLlama2Client)

    def review(self, section_name: str, section_content: str) -> str:
        """Review a section for clarity and readability."""
        chat_initiator = self.initiate_chat(
            self,
            message=f"""Please review this {section_name} section for clarity and readability.
            Provide ONLY feedback and suggestions. Do not rewrite the content.
            
            Guidelines:
            - Provide 3-5 key issues or suggestions
            - Be specific and actionable
            - Avoid repetition
            - Keep each point concise
            
            Section to review:
            {section_content}
            
            TERMINATE"""
        )
        return chat_initiator.last_message().get("content", "")

class CriticalAnalysisAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="critical_analysis_agent",
            system_message="""You are a critical analysis specialist. Your role is to:
            1. Evaluate the logical flow and argumentation
            2. Identify potential biases or assumptions
            3. Assess the strength of evidence and conclusions
            4. Suggest improvements for analytical depth
            
            Provide ONLY feedback and suggestions. Do not rewrite or modify the content.
            Focus on identifying areas that need stronger analysis.
            
            Guidelines for your response:
            - Provide 3-5 key analytical issues or suggestions
            - Be specific about logical gaps or weak arguments
            - Avoid repetition
            - Focus on critical analysis only
            - Keep each point concise""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomLlama2Client)

    def review(self, section_name: str, section_content: str) -> str:
        """Review a section for critical analysis."""
        chat_initiator = self.initiate_chat(
            self,
            message=f"""Please provide critical analysis of this {section_name} section.
            Provide ONLY feedback and suggestions. Do not rewrite the content.
            
            Guidelines:
            - Provide 3-5 key issues or suggestions
            - Be specific and actionable
            - Avoid repetition
            - Keep each point concise
            
            Section to review:
            {section_content}
            
            TERMINATE"""
        )
        return chat_initiator.last_message().get("content", "")

class ModeratorAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="moderator_agent",
            system_message="""You are the moderator responsible for:
            1. Synthesizing feedback from all specialists
            2. Resolving conflicts between different suggestions
            3. Prioritizing improvements
            4. Providing a summary of key changes needed
            
            Provide ONLY feedback and suggestions. Do not rewrite or modify the content.
            Focus on summarizing key issues and suggesting improvements.
            
            Guidelines for your response:
            - Provide 5-7 key prioritized improvements
            - Focus on the most important issues
            - Avoid repetition
            - Keep each point concise
            - Prioritize based on impact and feasibility
            
            IMPORTANT: Your final points MUST be wrapped in *** markers and be in numbered format, like this:
            ***
            1. First point
            2. Second point
            etc.
            ***""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomLlama2Client)

    def moderate(self, section_name: str, section_content: str, reviews: List[str]) -> str:
        """Synthesize reviews and provide guidance."""
        combined_reviews = "\n\n".join(reviews)
        chat_initiator = self.initiate_chat(
            self,
            message=f"""Please synthesize these reviews and provide guidance for the {section_name} section.
            Provide ONLY feedback and suggestions. Do not rewrite the content.
            
            Guidelines:
            - Provide 5-7 key prioritized improvements
            - Focus on the most important issues
            - Avoid repetition
            - Keep each point concise
            
            IMPORTANT: You MUST wrap your final points in *** markers and use numbered format.
            
            Original Section:
            {section_content}
            
            Reviews:
            {combined_reviews}
            
            Please provide your final prioritized points in this format:
            ***
            1. First point
            2. Second point
            etc.
            ***
            
            TERMINATE"""
        )
        return chat_initiator.last_message().get("content", "")

def review_section(section_name: str, section_content: str, full_report: Dict[str, str]) -> Dict:
    """
    Review a single section of the report while providing full context.
    
    Args:
        section_name (str): Name of the section being reviewed
        section_content (str): Content of the section being reviewed
        full_report (Dict[str, str]): Dictionary containing all sections for context
        
    Returns:
        Dict: Dictionary containing the review results
    """
    # Format the full report for context
    context_text = "\n\n".join([
        f"## {name}\n{content}" 
        for name, content in full_report.items()
    ])
    
    # Format the review message to include both the specific section and full context
    review_message_template = """Please review the {section_name} section, while considering the full report context.
    Focus only on providing feedback and suggestions for the {section_name} section.
    Do not rewrite the content.
    
    Guidelines:
    - Provide 3-5 key issues or suggestions
    - Be specific and actionable
    - Avoid repetition
    - Keep each point concise
    
    Full Report Context:
    {context_text}
    
    Section to review:
    {section_content}
    
    TERMINATE"""
    
    # Create agents with the configuration
    config_list = [
        {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "model_client_cls": "CustomLlama2Client",
            "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
            "n": 1,
            "params": {
                "max_new_tokens": 1000,
                "top_k": 50,
                "temperature": 0.1,
                "do_sample": True,
            },
        }
    ]
    
    llm_config = {
        "config_list": config_list,
        "cache_seed": None,
        "cache": None
    }
    
    # Initialize user proxy agent with termination message
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,  # Set to 0 to prevent auto-replies
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        llm_config=llm_config
    )
    
    # Initialize agents
    tech_agent = TechnicalAccuracyAgent(llm_config=llm_config)
    clarity_agent = ClarityAgent(llm_config=llm_config)
    crit_agent = CriticalAnalysisAgent(llm_config=llm_config)
    moderator = ModeratorAgent(llm_config=llm_config)
    
    # Register model client for all agents at once to avoid multiple warnings
    for agent in [user_proxy, tech_agent, clarity_agent, crit_agent, moderator]:
        agent.register_model_client(model_client_cls=CustomLlama2Client)
    
    # Get reviews from each agent with a single response
    reviews = []
    for agent in [tech_agent, clarity_agent, crit_agent]:
        print(f"\n{'='*50}")
        print(f"Getting review from {agent.name}...")
        print(f"{'='*50}")
        
        # Create a focused message for the agent
        review_message = review_message_template.format(
            section_name=section_name,
            context_text=context_text,
            section_content=section_content
        )
        
        # Use user proxy to initiate the chat and get a single response
        user_proxy.initiate_chat(
            agent,
            message=review_message,
            silent=False  # Show the interaction
        )
        
        # Get the last message from the agent
        review = agent.last_message().get("content", "")
        # Clean up any TERMINATE text from the response
        review = review.replace("TERMINATE", "").strip()
        reviews.append(review)
        print(f"\n{'='*50}")
        print(f"Review from {agent.name} completed")
        print(f"{'='*50}")
    
    # Create debate manager and add reviews
    debate_manager = DebateManager()
    for agent_name, review in zip(
        ["Technical Accuracy", "Clarity", "Critical Analysis"],
        reviews
    ):
        debate_manager.add_review(agent_name, review)
    
    # Get moderator's guidance using user proxy
    print(f"\n{'='*50}")
    print("Getting moderator's guidance...")
    print(f"{'='*50}")
    
    # Create a focused message for the moderator
    moderator_message = f"""Please provide your synthesis of these reviews for the {section_name} section.
    Focus only on summarizing key issues and suggesting improvements.
    Do not rewrite the content.
    
    Guidelines:
    - Provide 5-7 key prioritized improvements
    - Focus on the most important issues
    - Avoid repetition
    - Keep each point concise
    
    Original Section:
    {section_content}
    
    Reviews:
    {debate_manager.resolve()}
    
    TERMINATE"""
    
    # Use user proxy to initiate the chat and get a single response
    user_proxy.initiate_chat(
        moderator,
        message=moderator_message,
        silent=False  # Show the interaction
    )
    guidance = moderator.last_message().get("content", "")
    # Clean up any TERMINATE text from the guidance
    guidance = guidance.replace("TERMINATE", "").strip()
    
    # Add asterisks around the final points if they start with numbers
    if guidance.strip().startswith(('1.', '1 ', '1)')):
        guidance = "***\n" + guidance + "\n***"
    
    print(f"\n{'='*50}")
    print("Moderator's guidance completed")
    print(f"{'='*50}")
    
    # Add debug print before returning
    print("\nFinal guidance format:")
    print(guidance)
    
    return {
        "section_name": section_name,
        "original_content": section_content,
        "reviews": {
            "technical": reviews[0],
            "clarity": reviews[1],
            "critical": reviews[2],
        },
        "debate_summary": debate_manager.resolve(),
        "guidance": guidance
    }

def review_report(sections: Dict[str, str]) -> Dict:
    """
    Process each section of the report through the multi-agent review system.
    
    Args:
        sections (Dict[str, str]): Dictionary of section names and their content
        
    Returns:
        Dict: Dictionary containing the review results for each section
    """
    review_results = {}
    
    # Review each section separately but with full context
    for section_name, section_content in sections.items():
        print(f"\nReviewing {section_name} section...")
        review_results[section_name] = review_section(
            section_name, 
            section_content,
            sections  # Pass the full report dictionary
        )
    
    # Combine reviews and guidance into final report
    final_report = "\n\n".join([
        f"## {section_name}\n\n"
        f"### Original Content\n{result['original_content']}\n\n"
        f"### Reviews and Guidance\n{result['guidance']}"
        for section_name, result in review_results.items()
    ])
    
    return {
        "section_reviews": review_results,
        "final_report": final_report
    }

def load_report_from_json(paper_number: int) -> Dict:
    """Load a paper context from the litreviews folder."""
    litreviews_dir = Path("litreviews")
    paper_path = litreviews_dir / f"paper_{paper_number}.json"
    
    if not paper_path.exists():
        raise FileNotFoundError(f"No paper file found at {paper_path}")
    
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        return report_data
    except Exception as e:
        print(f"Error loading paper file from {paper_path}: {e}")
        raise

def save_review_results(results: Dict, output_dir: Path):
    """Save final points to JSON file."""
    print(f"\nAttempting to save results to {output_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified at {output_dir}")
    
    # Dictionary to store all sections' points
    all_sections_points = {}
    
    # Extract and save final points to JSON file
    for section_name, section_result in results["section_reviews"].items():
        print(f"\nProcessing section: {section_name}")
        guidance = section_result.get("guidance", "")
        print(f"Found guidance of length: {len(guidance)}")
        
        # Split by *** markers and get the last set of points
        parts = guidance.split("***")
        print(f"Found {len(parts)} parts after splitting by ***")
        
        # Look for the last set of numbered points between *** markers
        numbered_points = None
        for i in range(len(parts)-2, -1, -1):  # Search backwards through parts
            content = parts[i].strip()
            if content and content[0].isdigit() and '. ' in content:
                numbered_points = content
                break
        
        if numbered_points:
            print(f"Found numbered points: {numbered_points[:100]}...")
            
            # Extract numbered points
            points = []
            for line in numbered_points.split('\n'):
                line = line.strip()
                if line and line[0].isdigit() and '. ' in line:
                    points.append(line)
            print(f"Extracted {len(points)} points")
            
            if points:
                # Add points to the dictionary
                all_sections_points[section_name] = points
                print(f"Added {len(points)} points for section {section_name}")
    
    print(f"\nTotal sections with points: {len(all_sections_points)}")
    if all_sections_points:
        # Save all points to a JSON file
        points_path = output_dir / "final_points.json"
        try:
            with open(points_path, 'w', encoding='utf-8') as f:
                json.dump(all_sections_points, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved points to {points_path}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
    else:
        print("No points found to save!")

def main():
    # Look for paper files in the litreviews directory
    litreviews_dir = Path("litreviews")
    
    # Check for papers 1 through 20
    paper_numbers = []
    for i in range(1, 21):
        if (litreviews_dir / f"paper_{i}.json").exists():
            paper_numbers.append(i)
    
    if not paper_numbers:
        print("No paper files found in the litreviews directory!")
        return
    
    print(f"Found papers with numbers: {paper_numbers}")
    
    for paper_number in paper_numbers:
        print(f"\n{'='*70}")
        print(f"Processing paper {paper_number}")
        print(f"{'='*70}")
        
        # Create output directory for this paper
        output_dir = Path("output")
        paper_output_dir = output_dir / f"paper_{paper_number}"
        paper_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the report from JSON
        try:
            report_data = load_report_from_json(paper_number)
            
            # Check if the report has a 'sections' key
            if 'sections' not in report_data:
                print(f"Error: No 'sections' key found in paper {paper_number}.")
                print(f"Available keys: {list(report_data.keys())}")
                continue
                
            # Extract the sections from the nested structure
            sections = report_data['sections']
            
            print(f"Loaded report with {len(sections)} sections: {list(sections.keys())}")
            
            # Define the order of sections to process
            section_order = ["INTRODUCTION", "METHODOLOGY", "RESULTS", "DISCUSSION", "CONCLUSION"]
            
            # Dictionary to store all results
            all_results = {}
            
            # Process each section in order
            for section_name in section_order:
                if section_name in sections:
                    print(f"\n{'='*70}")
                    print(f"Processing {section_name} section...")
                    print(f"{'='*70}")
                    
                    # Create a single-section dictionary for processing
                    section_dict = {section_name: sections[section_name]}
                    results = review_report(section_dict)
                    
                    # Store the results
                    all_results.update(results["section_reviews"])
                    
                    print(f"\n{section_name} review completed successfully!")
                else:
                    print(f"\nWarning: {section_name} section not found in the report.")
            
            # Save all results
            if all_results:
                # Save to the output directory for this paper
                final_results = {
                    "section_reviews": all_results,
                    "final_report": "\n\n".join([
                        f"## {section_name}\n\n"
                        f"### Original Content\n{result['original_content']}\n\n"
                        f"### Reviews and Guidance\n{result['guidance']}"
                        for section_name, result in all_results.items()
                    ])
                }
                
                save_review_results(final_results, paper_output_dir)
                print(f"\nResults saved successfully to {paper_output_dir}!")
            else:
                print(f"\nError: No sections were processed successfully for paper {paper_number}.")
                print("Available sections:", list(sections.keys()))
                
        except Exception as e:
            print(f"Error processing paper {paper_number}: {e}")
            continue

if __name__ == "__main__":
    main()