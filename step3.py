import autogen
from typing import Dict, List, Any
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
import logging
from final_evaluation import calculate_technical_depth, calculate_clarity, calculate_structure, evaluate_citation_accuracy
import gc
import traceback
from rewrite_function import rewrite_text, load_shared_model, CustomGemmaClient  # Import necessary functions from rewrite_function

# Add class method for cleanup to CustomGemmaClient
# This extends the imported class with the needed cleanup method
def cleanup_shared_model(cls):
    """Clean up shared model resources"""
    # Clear CUDA cache and collect garbage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Shared model resources cleaned up")

# Add the cleanup method to the imported class
CustomGemmaClient.cleanup = classmethod(cleanup_shared_model)

def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"improvement_process_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logging()

# Load environment variables
load_dotenv()

# Clear CUDA cache and set memory management
torch.cuda.empty_cache()
torch.backends.cuda.max_memory_split_size = 256 * 1024 * 1024  # 256MB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'

# Set up the model configuration
os.environ["OAI_CONFIG_LIST"] = json.dumps(
    [
        {
            "model": "google/gemma-3-27b-it",  # Updated to use Gemma model
            "model_client_cls": "CustomGemmaClient",
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

def extract_json_from_response(response: str) -> Dict:
    """Helper function to extract JSON from agent responses."""
    if response.startswith("TERMINATE:"):
        response = response.replace("TERMINATE:", "").strip()
    
    if "```json" in response and "```" in response:
        json_str = response.split("```json")[1].split("```")[0].strip()
    else:
        if ": {" in response:
            json_str = response.split(": {", 1)[1].strip()
            json_str = "{" + json_str
        else:
            json_start = response.rfind('{')
            if json_start != -1:
                json_str = response[json_start:]
            else:
                json_str = response.strip()
    
    if "You MUST format your response as a JSON object" in json_str:
        json_str = json_str.split("You MUST format your response as a JSON object")[-1].strip()
    
    json_str = re.sub(r'^[^{]*', '', json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            clean_str = json_str.split("TERMINATE")[-1].strip()
            if clean_str.startswith(":"):
                clean_str = clean_str[1:].strip()
            return json.loads(clean_str)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response
            }

class TechnicalAccuracyAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="technical_accuracy_agent",
            system_message="""You are a technical accuracy specialist. Your role is to review content and provide a list of key improvements.

DO NOT use JSON format. DO NOT provide explanations before or after the list.
ONLY provide a numbered list of 5-7 specific, actionable improvements between *** markers.

Example format:
***
1. First improvement suggestion
2. Second improvement suggestion
3. Third improvement suggestion
4. Fourth improvement suggestion
5. Fifth improvement suggestion
***

Focus on technical accuracy, factual correctness, and technical depth.
Keep suggestions specific, actionable, and prioritized.""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomGemmaClient)

    def review(self, section_name: str, section_content: str) -> Dict:
        """Review a section for technical accuracy."""
        prompt = f"""Review this {section_name} section for technical accuracy.
Provide a numbered list of 5-7 specific, actionable improvements between *** markers.
Focus on factual accuracy, technical depth, and terminology usage.

Section to review:
{section_content}

TERMINATE"""

        self.initiate_chat(self, message=prompt)
        response = self.last_message().get("content", "").strip()
        
        try:
            # Extract improvements between *** markers
            if "***" in response:
                parts = response.split("***")
                if len(parts) >= 3:
                    improvements = parts[1].strip()
                    return {"improvements": improvements}
            
            # Fall back to the entire response if no *** markers
            return {"improvements": response}
        except Exception as e:
            print(f"Error parsing technical accuracy response: {str(e)}")
            return {"improvements": "Error occurred during technical analysis", "error": str(e)}

class ClarityAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="clarity_readability_agent",
            system_message="""You are a clarity and readability specialist. Your role is to review content and provide a list of key improvements.

DO NOT use JSON format. DO NOT provide explanations before or after the list.
ONLY provide a numbered list of 5-7 specific, actionable improvements between *** markers.

Example format:
***
1. First improvement suggestion
2. Second improvement suggestion
3. Third improvement suggestion
4. Fourth improvement suggestion
5. Fifth improvement suggestion
***

Focus on clarity, readability, flow, and structure.
Keep suggestions specific, actionable, and prioritized.""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomGemmaClient)

    def review(self, section_name: str, section_content: str) -> Dict:
        """Review a section for clarity and readability."""
        prompt = f"""Review this {section_name} section for clarity and readability.
Provide a numbered list of 5-7 specific, actionable improvements between *** markers.
Focus on improving clarity, readability, flow, and structure.

Section to review:
{section_content}

TERMINATE"""

        self.initiate_chat(self, message=prompt)
        response = self.last_message().get("content", "").strip()
        
        try:
            # Extract improvements between *** markers
            if "***" in response:
                parts = response.split("***")
                if len(parts) >= 3:
                    improvements = parts[1].strip()
                    return {"improvements": improvements}
            
            # Fall back to the entire response if no *** markers
            return {"improvements": response}
        except Exception as e:
            print(f"Error parsing clarity review response: {str(e)}")
            return {"improvements": "Error occurred during clarity review", "error": str(e)}

class StructureAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="structure_agent",
            system_message="""You are a structural analysis specialist. Your role is to review content and provide a list of key improvements focused on flow, coherence, and organization.

DO NOT use JSON format. DO NOT provide explanations before or after the list.
ONLY provide a numbered list of 5-7 specific, actionable improvements between *** markers.

Example format:
***
1. First improvement suggestion
2. Second improvement suggestion
3. Third improvement suggestion
4. Fourth improvement suggestion
5. Fifth improvement suggestion
***

Focus on document structure, paragraph flow, logical progression, transitions between ideas, and overall coherence.
Keep suggestions specific, actionable, and prioritized.""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomGemmaClient)

    def review(self, section_name: str, section_content: str) -> Dict:
        """Review a section for structure, flow and coherence."""
        prompt = f"""Review this {section_name} section for structure, flow, and coherence.
Provide a numbered list of 5-7 specific, actionable improvements between *** markers.
Focus on improving document organization, paragraph flow, logical progression, transitions between ideas, and overall coherence.

Section to review:
{section_content}

TERMINATE"""

        self.initiate_chat(self, message=prompt)
        response = self.last_message().get("content", "").strip()
        
        try:
            # Extract improvements between *** markers
            if "***" in response:
                parts = response.split("***")
                if len(parts) >= 3:
                    improvements = parts[1].strip()
                    return {"improvements": improvements}
            
            # Fall back to the entire response if no *** markers
            return {"improvements": response}
        except Exception as e:
            print(f"Error parsing structure review response: {str(e)}")
            return {"improvements": "Error occurred during structure review", "error": str(e)}

class ModeratorAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="moderator_agent",
            system_message="""You are the moderator responsible for synthesizing feedback from specialist agents.
Your task is to:
1. Analyze reviews from all specialist agents (technical accuracy, clarity, structure, and fact-checking)
2. Identify the most important improvements needed
3. Create a NEW consolidated list of suggested improvements
4. Ensure suggestions are clear, specific, and actionable
5. PAY SPECIAL ATTENTION to fact-checking feedback and citation accuracy - ALWAYS include ALL points from the fact-checking agent

IMPORTANT:
- Do NOT write replacements (e.g., do not say "Replace X with Y")
- Instead, write suggestions for improvement (e.g., "Improve X by clarifying...")
- Create NEW synthesized improvements that incorporate insights from all agents
- Include ALL fact-checking points without exception
- Prioritize citation accuracy and fact-checking concerns above all other improvements
- ALWAYS enclose your list with *** markers
- ALWAYS number your points

Example format:
***
1. Improve the technical accuracy by clarifying the quantum tunneling mechanism
2. Enhance the explanation of carrier dynamics by adding specific examples
3. Strengthen the connection between concepts X and Y
4. Add more detail about the impact of Z on device performance
5. Clarify the relationship between A and B
***""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomGemmaClient)
        self.logger = logging.getLogger(__name__)

    def review(self, section_name: str, section_content: str, reviews: Dict[str, str]) -> Dict:
        """Synthesize reviews and provide guidance."""
        try:
            if not reviews:
                self.logger.warning("No reviews provided for moderation")
                return {
                    "error": "No reviews provided",
                    "improvements": "No reviews available to synthesize"
                }
            
            # Extract all fact-checking points first
            fact_checking_points = []
            if "fact_checking_agent" in reviews:
                fact_check_review = reviews.get("fact_checking_agent", {})
                if isinstance(fact_check_review, dict) and "improvements" in fact_check_review:
                    fact_checking_content = fact_check_review.get("improvements", "")
                    # Extract points between *** markers if they exist
                    if "***" in fact_checking_content:
                        parts = fact_checking_content.split("***")
                        if len(parts) >= 3:
                            fact_checking_points_text = parts[1].strip()
                            # Parse numbered list
                            for line in fact_checking_points_text.split('\n'):
                                line = line.strip()
                                if line and re.match(r'^\d+\.', line):
                                    fact_checking_points.append(f"[CITATION CHECK] {line}")
            
            # Format the reviews more clearly
            reviews_text = ""
            for agent_name, review in reviews.items():
                if isinstance(review, dict):
                    review_content = review.get("improvements", "")
                else:
                    review_content = str(review)
                reviews_text += f"\n=== {agent_name.upper()} REVIEW ===\n{review_content}\n"
            
            message = f"""Please synthesize these reviews for the {section_name} section.
IMPORTANT: You MUST include ALL fact-checking points in your final list without exception.

SECTION: {section_name}

ORIGINAL CONTENT:
{section_content}

SPECIALIST REVIEWS:
{reviews_text}

FACT-CHECKING POINTS TO INCLUDE (THESE MUST ALL BE INCLUDED):
{chr(10).join(fact_checking_points)}

REQUIREMENTS:
1. Include ALL fact-checking points listed above
2. Add additional synthesized improvements from other specialists
3. Make each improvement specific and actionable
4. Prioritize citation accuracy above all other improvements
5. ALWAYS enclose your list with *** markers
6. ALWAYS number your points

Example format:
***
1. First synthesized improvement
2. Second synthesized improvement
etc.
***

TERMINATE"""
            
            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                code_execution_config=False
            )
            
            user_proxy.initiate_chat(
                self,
                message=message,
                silent=False
            )
            
            response = self.last_message().get("content", "")
            
            # Extract improvements between *** markers
            improvements = extract_last_asterisk_section(response)
            if not improvements:
                self.logger.warning("No improvements found between *** markers")
                improvements = response
            
            # Ensure all fact-checking points are included
            if fact_checking_points:
                if not all(point.lower() in improvements.lower() for point in fact_checking_points):
                    self.logger.warning("Not all fact-checking points were included in the moderator's response")
                    # Append any missing fact-checking points
                    missing_points = []
                    for point in fact_checking_points:
                        if point.lower() not in improvements.lower():
                            missing_points.append(point)
                    
                    if missing_points:
                        if not improvements.endswith('\n'):
                            improvements += '\n'
                        improvements += "\n# CRITICAL CITATION ACCURACY POINTS:\n"
                        for i, point in enumerate(missing_points, 1):
                            improvements += f"{i}. {point}\n"
            
            return {
                "improvements": improvements,
                "raw_response": response,
                "metadata": {
                    "total_reviews": len(reviews),
                    "fact_checking_points_included": len(fact_checking_points)
                }
            }
                
        except Exception as e:
            self.logger.error(f"Unexpected error in moderation: {str(e)}")
            return {
                "error": str(e),
                "improvements": "Error occurred during moderation",
                "metadata": {
                    "total_reviews": len(reviews)
                }
            }

def save_consolidated_output(data: Dict, section_name: str) -> str:
    """Save all output data to a single consolidated JSON file."""
    try:
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Ensure the data structure includes fact-checking results
        if "iterations" in data:
            for iteration in data["iterations"]:
                if "reviews" in iteration:
                    # Make sure fact-checking results are properly formatted
                    if "fact_checking_agent" in iteration["reviews"]["raw_reviews"]:
                        fact_check_review = iteration["reviews"]["raw_reviews"]["fact_checking_agent"]
                        if isinstance(fact_check_review, dict) and "improvements" in fact_check_review:
                            # Already properly formatted
                            pass
                        else:
                            # Format the review if it's not already a dict
                            iteration["reviews"]["raw_reviews"]["fact_checking_agent"] = {
                                "improvements": str(fact_check_review)
                            }
                
                # Ensure quality assessment includes citation metrics
                if "quality_assessment" in iteration:
                    if "metrics" in iteration["quality_assessment"]:
                        if "citation_accuracy" not in iteration["quality_assessment"]["metrics"]:
                            iteration["quality_assessment"]["metrics"]["citation_accuracy"] = {
                                "score": 0.0,
                                "details": "Citation accuracy not evaluated in this iteration"
                            }
                        else:
                            # Ensure citation analysis is preserved
                            citation_accuracy = iteration["quality_assessment"]["metrics"]["citation_accuracy"]
                            if "citation_analysis" in citation_accuracy:
                                # Add a summary of citation scores for easy reference
                                citation_summary = {}
                                for citation in citation_accuracy["citation_analysis"]:
                                    citation_id = citation.get("citation_id", "unknown")
                                    citation_summary[citation_id] = {
                                        "score": citation.get("score", 0.0),
                                        "needs_improvement": citation.get("score", 0.0) < 0.7,
                                        "justification_summary": citation.get("justification", "No justification provided")[:100] + "..." if len(citation.get("justification", "")) > 100 else citation.get("justification", "")
                                    }
                                citation_accuracy["citation_summary"] = citation_summary
        
        # Also ensure final result has citation data
        if "final_result" in data and "final_quality_assessment" in data["final_result"]:
            final_assessment = data["final_result"]["final_quality_assessment"]
            if "metrics" in final_assessment and "citation_accuracy" in final_assessment["metrics"]:
                citation_accuracy = final_assessment["metrics"]["citation_accuracy"]
                # Add summary for final result too
                citation_summary = {}
                for citation in citation_accuracy.get("citation_analysis", []):
                    citation_id = citation.get("citation_id", "unknown")
                    citation_summary[citation_id] = {
                        "score": citation.get("score", 0.0),
                        "needs_improvement": citation.get("score", 0.0) < 0.7,
                        "justification_summary": citation.get("justification", "No justification provided")[:100] + "..." if len(citation.get("justification", "")) > 100 else citation.get("justification", "")
                    }
                citation_accuracy["citation_summary"] = citation_summary
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{section_name}_consolidated_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✅ Successfully saved consolidated output to: {str(output_path)}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error saving consolidated output: {e}")
        return "Error: Failed to save consolidated output"

def rewrite_section(section_text: str, improvements: List[str], model_config: Dict) -> str:
    """Direct LLM call to rewrite section based on improvements. No agents involved."""
    
    # Format the improvement points into a single string
    if isinstance(improvements, list):
        improvement_points = improvements
    else:
        # If improvements is a string, split it into a list if it's not already a list
        if isinstance(improvements, str):
            improvement_points = [improvements.strip()]
        else:
            improvement_points = []
    
    print(f"Calling rewrite_text function with {len(improvement_points)} improvement points")
    
    # Use the imported rewrite_text function - no fallback to original implementation
    return rewrite_text(
        improvement_points=improvement_points,
        original_text=section_text,
        temperature=0.7,
        max_tokens=2000
    )

def assess_quality(improved_text: str, original_text: str, referenced_papers: Dict = None, full_report_context: str = None) -> Dict:
    """Perform a quality assessment of the improved text using evaluation metrics."""
    
    # Calculate metrics for improved version
    metrics = {
        "technical_depth": calculate_technical_depth(improved_text),
        "clarity": calculate_clarity(improved_text),
        "structure": calculate_structure(full_report_context if full_report_context else improved_text)
    }
    
    # Add citation accuracy metrics if referenced papers are provided
    if referenced_papers:
        citation_accuracy = evaluate_citation_accuracy(improved_text, referenced_papers)
        metrics["citation_accuracy"] = {
            "score": citation_accuracy["score"],
            "citation_analysis": citation_accuracy["citation_analysis"],
            "needs_improvement": citation_accuracy["needs_improvement"],
            "improvement_suggestions": citation_accuracy["improvement_suggestions"]
        }
    
    # Check citation preservation
    original_citations = set(re.findall(r'\[\d+\]', original_text))
    improved_citations = set(re.findall(r'\[\d+\]', improved_text))
    citations_preserved = original_citations.issubset(improved_citations)
    
    # Log detailed metrics
    logger.info("\nQuality Assessment Details:")
    logger.info("\nTechnical Depth:")
    logger.info(f"- Combined Score: {metrics['technical_depth']['combined_score']:.3f}")
    logger.info(f"- Technical Terms Score: {metrics['technical_depth']['technical_term_metrics']['normalized_score']:.3f}")
    logger.info(f"- Concept Hierarchy: {metrics['technical_depth']['concept_hierarchy_depth']:.3f}")
    if 'llm_evaluation' in metrics['technical_depth']:
        logger.info(f"- LLM Score: {metrics['technical_depth']['llm_evaluation']['score']:.3f}")
    
    logger.info("\nClarity:")
    logger.info(f"- Combined Score: {metrics['clarity']['combined_score']:.3f}")
    logger.info(f"- Flesch Score: {metrics['clarity']['flesch_score']:.3f}")
    logger.info(f"- Defined Terms: {metrics['clarity']['defined_terms_count']:.3f}")
    logger.info(f"- Examples: {metrics['clarity']['example_count']:.3f}")
    if 'llm_evaluation' in metrics['clarity']:
        logger.info(f"- LLM Score: {metrics['clarity']['llm_evaluation']['score']:.3f}")
    
    logger.info("\nStructure:")
    logger.info(f"- Combined Score: {metrics['structure']['combined_score']:.3f}")
    logger.info(f"- Coherence Flow: {metrics['structure']['coherence']['concept_flow']['flow_score']:.3f}")
    if 'llm_evaluation' in metrics['structure']:
        logger.info(f"- LLM Score: {metrics['structure']['llm_evaluation']['score']:.3f}")
    
    if 'citation_accuracy' in metrics:
        logger.info("\nCitation Accuracy:")
        logger.info(f"- Overall Score: {metrics['citation_accuracy']['score']:.3f}")
        logger.info("\nDetailed Citation Analysis:")
        for citation in metrics['citation_accuracy']['citation_analysis']:
            logger.info(f"  [Citation {citation['citation_id']}]:")
            logger.info(f"  - Score: {citation['score']:.3f}")
            logger.info(f"  - Justification: {citation['justification']}")
            if citation['score'] < 0.7:
                suggestions = metrics['citation_accuracy']['improvement_suggestions'].get(citation['citation_id'], {})
                if suggestions:
                    logger.info(f"  - Suggestion: {suggestions.get('suggestion', 'No specific suggestion')}")
            logger.info("")
    
    return {
        "metrics": metrics,
        "citations": {
            "preserved": citations_preserved,
            "missing": list(original_citations - improved_citations) if not citations_preserved else []
        },
        "length_ratio": len(improved_text) / len(original_text) if len(original_text) > 0 else 0
    }

def extract_last_asterisk_section(text: str) -> str:
    """Extract the last section enclosed in *** markers from a text."""
    if "***" not in text:
        return ""
    
    parts = text.split("***")
    if len(parts) < 3:
        return ""
    
    return parts[-2].strip()

def parse_improvement_points(text: str) -> List[str]:
    """Parse numbered improvement points from text."""
    points = []
    current_point = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a new numbered point
        if re.match(r'^\d+[\.\)\-:] ', line):
            if current_point:
                points.append(' '.join(current_point))
                current_point = []
            current_point.append(line)
        elif current_point:
            current_point.append(line)
            
    # Add the last point if exists
    if current_point:
        points.append(' '.join(current_point))
        
    return points

def get_moderator_improvements(review_results: Dict) -> List[str]:
    """Extract improvement points from moderator's review."""
    try:
        # Get moderator's raw output from last message like other agents
        if isinstance(review_results, dict) and "moderator_agent" in review_results.get("raw_reviews", {}):
            moderator = ModeratorAgent()
            moderator_text = moderator.last_message().get("content", "")
        else:
            moderator_text = str(review_results)
            
        logger.debug("\nRAW MODERATOR OUTPUT:")
        logger.debug(moderator_text)
        
        if not moderator_text.strip():
            logger.warning("WARNING: Empty moderator text")
            return []
            
        # Extract content between *** markers
        points_section = extract_last_asterisk_section(moderator_text)
        if not points_section:
            logger.warning("No content found between *** markers")
            return []
            
        # Parse the points into a list
        points = parse_improvement_points(points_section)
        
        logger.info("\nEXTRACTED IMPROVEMENT POINTS:")
        for i, point in enumerate(points, 1):
            logger.info(f"{i}. {point}")
            
        return points
        
    except Exception as e:
        logger.error(f"Error extracting moderator improvements: {str(e)}")
        return []

def get_needed_agents(metric_results: Dict[str, bool]) -> List[str]:
    """Determine which agents are needed based on failed LLM metrics."""
    needed_agents = []
    
    # Technical depth agent needed if LLM evaluation failed
    if not metric_results['technical_depth']['overall']:
        logger.info("Technical depth agent needed: LLM evaluation below threshold")
        needed_agents.append('technical_accuracy_agent')
    
    # Clarity agent needed if LLM evaluation failed
    if not metric_results['clarity']['overall']:
        logger.info("Clarity agent needed: LLM evaluation below threshold")
        needed_agents.append('clarity_agent')
    
    # Structure agent needed if LLM evaluation failed
    if not metric_results['structure']['overall']:
        logger.info("Structure agent needed: LLM evaluation below threshold")
        needed_agents.append('structure_agent')
    
    # Citation accuracy agent needed if overall score or any individual citation is below threshold
    if not metric_results['citation_accuracy']['overall'] or \
       any(not citation_info['meets_threshold'] 
           for citation_info in metric_results['citation_accuracy']['individual_citations'].values()):
        logger.info("Fact checking agent needed: Citation accuracy issues found")
        if not metric_results['citation_accuracy']['overall']:
            logger.info("- Overall citation accuracy below threshold")
        
        # Log which citations need improvement
        for citation_id, info in metric_results['citation_accuracy']['individual_citations'].items():
            if not info['meets_threshold']:
                logger.info(f"- Citation [{citation_id}] below threshold (score: {info['score']:.2f})")
                logger.info(f"  Justification: {info['justification']}")
        
        needed_agents.append('fact_checking_agent')
    
    return list(dict.fromkeys(needed_agents))  # Remove duplicates while preserving order

def check_metric_thresholds(metrics: Dict) -> Dict[str, bool]:
    """Check if metrics meet defined thresholds using only combined scores for each metric type."""
    
    thresholds = {
        'technical_depth': {
            'combined_score': 0.7     # Combined score threshold only
        },
        'clarity': {
            'combined_score': 0.7     # Combined score threshold only
        },
        'structure': {
            'combined_score': 0.7     # Combined score threshold only
        },
        'citation_accuracy': {
            'overall_score': 0.8,      # At least 80% overall citation accuracy
            'individual_threshold': 0.7  # At least 70% for each individual citation
        }
    }
    
    results = {
        'technical_depth': {
            'overall': False
        },
        'clarity': {
            'overall': False
        },
        'structure': {
            'overall': False
        },
        'citation_accuracy': {
            'overall': False,
            'individual_citations': {}
        }
    }
    
    # Check technical depth metrics - only use combined score
    tech_metrics = metrics['technical_depth']
    results['technical_depth']['overall'] = (
        tech_metrics.get('combined_score', 0) >= thresholds['technical_depth']['combined_score']
    )
    
    # Check clarity metrics - only use combined score
    clarity_metrics = metrics['clarity']
    results['clarity']['overall'] = (
        clarity_metrics.get('combined_score', 0) >= thresholds['clarity']['combined_score']
    )
    
    # Check structure metrics - only use combined score
    structure_metrics = metrics['structure']
    results['structure']['overall'] = (
        structure_metrics.get('combined_score', 0) >= thresholds['structure']['combined_score']
    )
    
    # Check citation accuracy metrics if available
    if 'citation_accuracy' in metrics:
        citation_metrics = metrics['citation_accuracy']
        
        # Check overall citation accuracy
        results['citation_accuracy']['overall'] = citation_metrics.get('score', 0) >= thresholds['citation_accuracy']['overall_score']
        
        # Check individual citations
        for citation in citation_metrics.get('citation_analysis', []):
            citation_id = citation['citation_id']
            citation_score = citation['score']
            results['citation_accuracy']['individual_citations'][citation_id] = {
                'meets_threshold': citation_score >= thresholds['citation_accuracy']['individual_threshold'],
                'score': citation_score,
                'context': citation.get('contexts', []),
                'justification': citation.get('justification', '')
            }
    
    return results

def selective_review_section(section_name: str, section_content: str, needed_agents: List[str], referenced_papers: Dict = None, previous_citation_scores: Dict = None) -> Dict:
    """Review section using only specified agents."""
    config_list = [
        {
            "model": "google/gemma-3-27b-it",  # Updated to use Gemma model
            "model_client_cls": "CustomGemmaClient",
            "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
            "n": 1,
            "params": {
                "max_new_tokens": 2000,
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
    
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        llm_config=llm_config
    )
    
    # Initialize only needed agents
    agents = {}
    if 'technical_accuracy_agent' in needed_agents:
        agents['technical_accuracy_agent'] = TechnicalAccuracyAgent(llm_config=llm_config)
    if 'clarity_agent' in needed_agents:
        agents['clarity_agent'] = ClarityAgent(llm_config=llm_config)
    if 'structure_agent' in needed_agents:
        agents['structure_agent'] = StructureAgent(llm_config=llm_config)
    if 'fact_checking_agent' in needed_agents:
        agents['fact_checking_agent'] = FactCheckingAgent(llm_config=llm_config)
    
    # Always include moderator
    moderator = ModeratorAgent(llm_config=llm_config)
    
    # Register model client for all agents
    for agent in [user_proxy] + list(agents.values()) + [moderator]:
        agent.register_model_client(model_client_cls=CustomGemmaClient)
    
    reviews = {}
    cleaned_reviews = {}
    
    # Run reviews from selected agents
    for agent_name, agent in agents.items():
        print(f"\n{'='*50}")
        print(f"Getting review from {agent_name}...")
        print(f"{'='*50}")
        
        if agent_name == 'fact_checking_agent' and referenced_papers:
            # Build reference content string for fact checking
            reference_content = "\nREFERENCED PAPERS CONTENT:\n"
            for ref_title, ref_data in referenced_papers.items():
                citation_id = ref_data.get('citation_id', 'unknown')
                reference_content += f"\n[{citation_id}] {ref_title}\n"
                reference_content += f"Abstract: {ref_data.get('abstract', 'No abstract available')}\n"
                if ref_data.get('chunks'):
                    reference_content += "Content chunks:\n"
                    for chunk in ref_data['chunks']:
                        reference_content += f"{chunk}\n"
                    reference_content += "-" * 80 + "\n"
                    
            # Standard review for fact checking agent with reference data
            review_message = f"""Review this {section_name} section and verify if the content is supported by the reference data.
            Provide a numbered list of 5-7 specific, actionable improvements between *** markers.
            Focus on factual accuracy and alignment with the provided references.
            
            Section to review:
            {section_content}
            
            Reference data:
            {reference_content}
            
            TERMINATE"""
            
            user_proxy.initiate_chat(
                agent,
                message=review_message,
                silent=False
            )
            review_result = agent.last_message().get("content", "")
            
            # No validation or retry attempt - accept the response as is
            reviews[agent_name] = review_result
            cleaned_reviews[agent_name] = extract_last_asterisk_section(str(review_result)) or str(review_result)
        else:
            # Standard review for other agents
            review_message = f"""Please review this {section_name} section.
            Focus on providing feedback as a numbered list of 5-7 improvements between *** markers.
            
            Guidelines:
            - Provide 5-7 key issues or suggestions
            - Be specific and actionable
            - Avoid repetition
            - Keep each point concise
            - Use *** markers before and after your list
            
            Section to review:
            {section_content}
            
            TERMINATE"""
            
            user_proxy.initiate_chat(
                agent,
                message=review_message,
                silent=False
            )
            review_result = agent.last_message().get("content", "")
        
        reviews[agent_name] = review_result
        cleaned_reviews[agent_name] = extract_last_asterisk_section(str(review_result)) or str(review_result)
    
    # Get moderator's guidance
    debate_manager = DebateManager()
    for agent_name, review in cleaned_reviews.items():
        debate_manager.add_review(agent_name, review)
    
    moderator_message = f"""Please synthesize these reviews for the {section_name} section.
    Focus specifically on improvements related to: {', '.join(needed_agents)}.
    Output ONLY a numbered list of 5-7 specific, actionable improvements between *** markers.
    
    Original Section:
    {section_content}
    
    Reviews to synthesize:
    {debate_manager.resolve()}
    
    TERMINATE"""
    
    user_proxy.initiate_chat(
        moderator,
        message=moderator_message,
        silent=False
    )
    
    raw_guidance = moderator.last_message().get("content", "").strip()
    cleaned_guidance = extract_last_asterisk_section(raw_guidance) or raw_guidance
    
    reviews["moderator_agent"] = raw_guidance
    cleaned_reviews["moderator_agent"] = cleaned_guidance
    
    return {
        "section_name": section_name,
        "original_content": section_content,
        "raw_reviews": reviews,
        "cleaned_reviews": cleaned_reviews
    }

def get_all_chapter_files() -> List[Path]:
    """Get all JSON files from the initial_chapters directory."""
    chapters_dir = Path("initial_chapters")
    if not chapters_dir.exists():
        raise FileNotFoundError(f"Directory {chapters_dir} not found")
    return sorted(chapters_dir.glob("chapter_*.json"))

def create_chapter_markdown(chapter_number: int, sections: dict, consolidated_outputs: dict) -> None:
    """Create a markdown file for a chapter combining all sections and references."""
    output_dir = Path("chapter_markdowns")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"chapter_{chapter_number}.md"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write chapter header
            f.write(f"# Chapter {chapter_number}\n\n")
            
            # Process each section in order
            for section_name, original_content in sections.items():
                f.write(f"## {section_name}\n\n")
                
                # Check if this is a references section
                if section_name.upper() in ['REFERENCES', 'REFERENCE', 'BIBLIOGRAPHY']:
                    # Write preserved references without any modifications
                    f.write(f"{original_content}\n\n")
                else:
                    # Get final improved content from consolidated output
                    section_output = consolidated_outputs.get(section_name, {})
                    final_result = section_output.get("final_result", {})
                    
                    # Add citation accuracy metrics if available and not a references section
                    if "quality_assessment" in final_result and "metrics" in final_result["quality_assessment"]:
                        citation_metrics = final_result["quality_assessment"]["metrics"].get("citation_accuracy", {})
                        if citation_metrics:
                            f.write(f"### Citation Accuracy Analysis\n\n")
                            f.write(f"**Overall Score: {citation_metrics.get('score', 0):.2f}**\n\n")
                            
                            # Add detailed citation analysis
                            f.write("#### Individual Citation Analysis\n\n")
                            for citation in citation_metrics.get('citation_analysis', []):
                                score_color = "green" if citation['score'] >= 0.7 else "orange" if citation['score'] >= 0.4 else "red"
                                f.write(f"- **Citation [{citation['citation_id']}]**: Score: <span style='color:{score_color}'>{citation['score']:.2f}</span>\n")
                                f.write(f"  - **Justification**: {citation['justification']}\n")
                                
                                # Add improvement suggestions for citations below threshold
                                if citation['score'] < 0.7:
                                    suggestions = citation_metrics.get('improvement_suggestions', {}).get(citation['citation_id'], {})
                                    if suggestions:
                                        f.write(f"  - **Suggestion**: {suggestions.get('suggestion', 'No specific suggestion')}\n")
                                f.write("\n")
                            
                            f.write("---\n\n")
                    
                    final_content = final_result.get("final_text", original_content)
                    f.write(f"{final_content}\n\n")
        
        logger.info(f"Created markdown file for Chapter {chapter_number}: {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating markdown file for Chapter {chapter_number}: {e}")
        logger.error(traceback.format_exc())

class FactCheckingAgent(autogen.AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="fact_checking_agent",
            system_message="""You are a fact-checking specialist. Your task is to verify if the content is supported by the reference data provided.

DO NOT use JSON format. DO NOT provide explanations before or after the list.
ONLY provide a numbered list of 5-7 specific, actionable improvements between *** markers.

Example format:
***
1. First fact-checking improvement suggestion
2. Second fact-checking improvement suggestion
3. Third fact-checking improvement suggestion
4. Fourth fact-checking improvement suggestion
5. Fifth fact-checking improvement suggestion
***

Focus on:
- Ensuring claims in the text are supported by the reference data
- Identifying any potential inaccuracies or misrepresentations
- Suggesting improvements to align content with reference data
- Maintaining scientific/technical accuracy

Keep suggestions specific, actionable, and prioritized.""",
            **kwargs
        )
        self.register_model_client(model_client_cls=CustomGemmaClient)

    def review(self, section_name: str, section_content: str, referenced_papers: Dict = None) -> Dict:
        """Review a section for citation accuracy."""
        if not referenced_papers:
            return {
                "improvements": "No references provided for fact-checking.",
                "type": "fact_checking_agent",
                "has_references": False
            }

        # Build reference content string
        reference_content = "\nREFERENCED PAPERS CONTENT:\n"
        for ref_title, ref_data in referenced_papers.items():
            citation_id = ref_data.get('citation_id', 'unknown')
            reference_content += f"\n[{citation_id}] {ref_title}\n"
            reference_content += f"Abstract: {ref_data.get('abstract', 'No abstract available')}\n"
            if ref_data.get('chunks'):
                reference_content += "Content chunks:\n"
                for chunk in ref_data['chunks']:
                    reference_content += f"{chunk}\n"
                reference_content += "-" * 80 + "\n"

        prompt = f"""Review this {section_name} section and verify if the content is supported by the reference data.
Provide a numbered list of 5-7 specific, actionable improvements between *** markers.
Focus on factual accuracy and alignment with the provided references.

Section to review:
{section_content}

Reference data:
{reference_content}

TERMINATE"""

        self.initiate_chat(self, message=prompt)
        response = self.last_message().get("content", "").strip()
        
        try:
            # Extract improvements between *** markers
            if "***" in response:
                parts = response.split("***")
                if len(parts) >= 3:
                    improvements = parts[1].strip()
                    return {"improvements": improvements}
            
            # Fall back to the entire response if no *** markers
            return {"improvements": response}
        except Exception as e:
            print(f"Error parsing fact-checking response: {str(e)}")
            return {"improvements": "Error occurred during fact-checking", "error": str(e)}

def main():
    """Main function to run the iterative review and rewrite process for all chapters and sections."""
    try:
        # Get all chapter files
        chapter_files = get_all_chapter_files()
        if not chapter_files:
            logger.error("No chapter files found in initial_chapters directory")
            return
            
        logger.info(f"Found {len(chapter_files)} chapter files to process")
        
        # Initial memory cleanup
        CustomGemmaClient.cleanup()
        
        # Process each chapter
        for chapter_file in chapter_files:
            try:
                chapter_number = int(chapter_file.stem.split('_')[1])
                logger.info(f"\n{'='*100}")
                logger.info(f"Processing Chapter {chapter_number}")
                logger.info(f"{'='*100}")
                
                # Load chapter data
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                # Get all sections and referenced papers from this chapter
                sections = report_data.get('sections', {})
                referenced_papers = report_data.get('referenced_papers', {})
                
                if not sections:
                    logger.warning(f"No sections found in Chapter {chapter_number}")
                    continue
                    
                logger.info(f"Found {len(sections)} sections to process")
                
                # Dictionary to store consolidated outputs for this chapter
                chapter_consolidated_outputs = {}
                
                # Process each section
                for section_name, section_content in sections.items():
                    try:
                        # Skip references section (case-insensitive check)
                        if section_name.upper() in ['REFERENCES', 'REFERENCE', 'BIBLIOGRAPHY']:
                            logger.info(f"\nSkipping {section_name} section - preserving as is")
                            chapter_consolidated_outputs[section_name] = {
                                "metadata": {
                                    "chapter": chapter_number,
                                    "section": section_name,
                                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                                    "skipped": True,
                                    "reason": "References section preserved without modifications"
                                },
                                "original_content": section_content,
                                "final_result": {
                                    "final_text": section_content,
                                    "status": "preserved"
                                }
                            }
                            continue
                        
                        logger.info(f"\n{'='*80}")
                        logger.info(f"Processing Chapter {chapter_number} - Section: {section_name}")
                        logger.info(f"{'='*80}")
                        
                        # Skip empty sections
                        if not section_content or section_content.isspace():
                            logger.warning(f"Skipping empty section: {section_name}")
                            continue

                        # Create consolidated output structure for this section
                        consolidated_output = {
                            "metadata": {
                                "chapter": chapter_number,
                                "section": section_name,
                                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                                "model": "google/gemma-3-27b-it",
                                "skipped": False
                            },
                            "iterations": [],
                            "final_result": None
                        }

                        current_text = section_content
                        max_iterations = 3  # Allow up to 3 iterations
                        
                        # Store current quality assessment to avoid redundant evaluation
                        current_quality_assessment = None

                        # Iterative improvement process
                        for iteration in range(1, max_iterations + 1):
                            logger.info(f"\n{'='*50}")
                            logger.info(f"Processing iteration {iteration}/{max_iterations}")
                            logger.info(f"{'='*50}")
                            
                            # First iteration: Full review with all agents
                            if iteration == 1:
                                # Use all agents for first iteration
                                all_agents = ['technical_accuracy_agent', 'clarity_agent', 'structure_agent']
                                if referenced_papers:
                                    all_agents.append('fact_checking_agent')
                                review_results = selective_review_section(section_name, current_text, all_agents, referenced_papers)
                            else:
                                # Use quality assessment from previous iteration instead of recalculating
                                if current_quality_assessment:
                                    # Check metrics against thresholds
                                    metric_results = check_metric_thresholds(current_quality_assessment["metrics"])
                                    
                                    # Determine which agents are needed
                                    needed_agents = get_needed_agents(metric_results)
                                    
                                    if not needed_agents:
                                        logger.info(f"All quality thresholds met! No further iterations needed.")
                                        break
                                    
                                    logger.info(f"Agents needed for iteration {iteration}: {', '.join(needed_agents)}")
                                    
                                    # Previous citation scores for fact checking
                                    prev_citation_scores = current_quality_assessment["metrics"].get("citation_accuracy", None)
                                    
                                    # Selective review with only the needed agents
                                    review_results = selective_review_section(
                                        section_name, 
                                        current_text, 
                                        needed_agents, 
                                        referenced_papers,
                                        prev_citation_scores
                                    )
                                else:
                                    # Fallback - should not happen in normal operation
                                    logger.warning("No quality assessment from previous iteration, using all agents")
                                    all_agents = ['technical_accuracy_agent', 'clarity_agent', 'structure_agent']
                                    if referenced_papers:
                                        all_agents.append('fact_checking_agent')
                                    review_results = selective_review_section(section_name, current_text, all_agents, referenced_papers)
                            
                            # Extract improvement points from moderator's raw output
                            moderator_output = review_results.get("raw_reviews", {}).get("moderator_agent", "")
                            if isinstance(moderator_output, dict):
                                moderator_output = moderator_output.get("improvements", "")
                            improvement_points = get_moderator_improvements(moderator_output)
                            
                            if not improvement_points:
                                logger.warning(f"No improvement points extracted, ending iteration process")
                                break
                            
                            # Rewrite section with improvements
                            try:
                                # Format improvements as suggestions
                                improvements_text = "\n".join(f"- {point}" for point in improvement_points)
                                
                                # Create model config
                                config_list = [
                                    {
                                        "model": "google/gemma-3-27b-it",
                                        "model_client_cls": "CustomGemmaClient",
                                        "device": f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
                                        "n": 1,
                                        "params": {
                                            "max_new_tokens": 2000,
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
                                
                                improved_text = rewrite_section(
                                    current_text,
                                    improvements_text,
                                    llm_config
                                )
                                status = "improved"
                                
                                # Update current text for next iteration
                                previous_text = current_text
                                current_text = improved_text
                            except Exception as e:
                                logger.error(f"Error during rewrite: {e}")
                                improved_text = current_text
                                previous_text = current_text
                                status = "no_improvements_applied"
                            
                            # Assess quality with citation accuracy
                            current_quality_assessment = assess_quality(improved_text, section_content, referenced_papers, "\n\n".join(sections.values()))
                            
                            # Store iteration data
                            iteration_data = {
                                "iteration_number": iteration,
                                "reviews": review_results,
                                "improvement_points": improvement_points,
                                "text": {
                                    "before": previous_text,
                                    "after": improved_text
                                },
                                "quality_assessment": current_quality_assessment,
                                "status": status,
                                "referenced_papers_used": bool(referenced_papers)
                            }
                            
                            consolidated_output["iterations"].append(iteration_data)
                            
                            # Check if we've reached the target quality
                            metric_results = check_metric_thresholds(current_quality_assessment["metrics"])
                            if all(metric_results[category]['overall'] for category in ['technical_depth', 'clarity', 'structure']) and \
                               metric_results.get('citation_accuracy', {}).get('overall', True):
                                logger.info(f"All quality thresholds met after iteration {iteration}! No further iterations needed.")
                                break
                        
                        # Add final result to consolidated output
                        consolidated_output["final_result"] = {
                            "original_text": section_content,
                            "final_text": current_text,  # The latest version
                            "total_iterations": len(consolidated_output["iterations"]),
                            "final_status": "improved" if current_text != section_content else "unchanged",
                            "final_quality_assessment": current_quality_assessment,
                            "referenced_papers_used": bool(referenced_papers)
                        }
                        
                        # Store consolidated output for this section
                        output_path = save_consolidated_output(consolidated_output, f"chapter_{chapter_number}_{section_name}")
                        logger.info(f"\nChapter {chapter_number} - Section {section_name} completed after {len(consolidated_output['iterations'])} iterations!")
                        logger.info(f"Consolidated output saved to: {output_path}")
                        
                        # Add to chapter outputs
                        chapter_consolidated_outputs[section_name] = consolidated_output
                        
                    except Exception as e:
                        logger.error(f"Error processing section {section_name}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
                
                # Create markdown file for this chapter
                create_chapter_markdown(chapter_number, sections, chapter_consolidated_outputs)
                
                # Cleanup after chapter is done
                CustomGemmaClient.cleanup()
                logger.info(f"\nChapter {chapter_number} completed!")
                
            except Exception as e:
                logger.error(f"Error processing Chapter {chapter_number}: {str(e)}")
                CustomGemmaClient.cleanup()
                continue
        
        logger.info(f"\n{'='*100}")
        logger.info("All chapters processed successfully!")
        logger.info(f"{'='*100}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        CustomGemmaClient.cleanup()
        raise

if __name__ == "__main__":
    main()

