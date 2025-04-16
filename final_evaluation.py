import json
import textstat
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import anthropic
from sentence_transformers import SentenceTransformer
import nltk
from typing import Dict, List, Set
import graphviz
from gensim import corpora, models
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import openai

# Initialize sentence transformer at module level
try:
    from sentence_transformers import SentenceTransformer
    global_encoder = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    print("Warning: SentenceTransformer not available. Citation accuracy checks will be limited.")
    global_encoder = None

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key="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")

class ContextualCoherenceAnalyzer:
    def __init__(self):
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not initialize SentenceTransformer: {e}")
            self.encoder = None
            
    def analyze_contextual_coherence(self, text):
        """Analyze how ideas develop and connect throughout the text"""
        if not self.encoder or not text:
            return {
                'concept_flow': {'flow_score': 0, 'concept_chains': []}
            }
        
        # Clean and normalize text
        text = text.strip()
        if not text:
            return {
                'concept_flow': {'flow_score': 0, 'concept_chains': []}
            }
            
        # Split text into meaningful chunks (paragraphs or sections)
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        if not chunks:
            chunks = [text]  # Use whole text as one chunk if no clear splits
            
        # Ensure minimum content for analysis
        if len(chunks) < 2:
            # If single chunk is long enough, split it into sentences
            if len(text.split()) > 50:  # Minimum word threshold
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                if len(sentences) >= 2:
                    chunks = sentences
                else:
                    return {
                        'concept_flow': {
                            'flow_score': 0.5,  # Default score for very short content
                            'quality': 'limited content',
                            'details': {
                                'local_coherence': {'score': 0.5, 'assessment': 'insufficient content'},
                                'progression': {'score': 0.5, 'assessment': 'insufficient content'}
                            }
                        }
                    }
            else:
                return {
                    'concept_flow': {
                        'flow_score': 0.5,
                        'quality': 'limited content',
                        'details': {
                            'local_coherence': {'score': 0.5, 'assessment': 'insufficient content'},
                            'progression': {'score': 0.5, 'assessment': 'insufficient content'}
                        }
                    }
                }
        
        # Analyze concept flow with chunks
        concept_flow = self.analyze_concept_flow(chunks)
        
        return {
            'concept_flow': concept_flow
        }
        
    def analyze_concept_flow(self, chunks):
        """Analyze how concepts flow with improved robustness and error handling"""
        try:
            # Encode chunks
            embeddings = self.encoder.encode(chunks)
            
            # Calculate local coherence with error handling
            local_scores = []
            for i in range(len(chunks) - 1):
                try:
                    similarity = np.dot(embeddings[i], embeddings[i+1]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
                    # Handle potential NaN from zero division
                    if np.isnan(similarity):
                        similarity = 0.5  # Default to neutral score
                    local_scores.append(float(similarity))
                except Exception as e:
                    print(f"Warning: Error calculating local coherence: {e}")
                    local_scores.append(0.5)  # Default to neutral score
            
            if not local_scores:  # If no scores were calculated
                return {
                    'flow_score': 0.5,
                    'quality': 'calculation error',
                    'details': {
                        'local_coherence': {'score': 0.5, 'assessment': 'calculation error'},
                        'progression': {'score': 0.5, 'assessment': 'calculation error'}
                    }
                }
            
            # Calculate average local coherence
            avg_local = np.mean(local_scores)
            
            # Evaluate progression
            progression_scores = []
            for i in range(len(chunks) - 2):
                try:
                    score1 = local_scores[i]
                    score2 = local_scores[i + 1]
                    variation = abs(score1 - score2)
                    progression_scores.append(1.0 if 0.1 <= variation <= 0.4 else 0.5)
                except Exception as e:
                    print(f"Warning: Error calculating progression: {e}")
                    progression_scores.append(0.5)
            
            if not progression_scores:
                progression_scores = [0.5]  # Default if can't calculate progression
            
            # Calculate final score with safe averaging
            local_quality = min(max(avg_local, 0.0), 1.0)  # Clamp between 0 and 1
            progression_quality = min(max(np.mean(progression_scores), 0.0), 1.0)
            
            final_score = (
                0.4 * local_quality +
                0.6 * progression_quality
            )
            
            return {
                'flow_score': float(final_score),
                'quality': self.get_quality_label(final_score),
                'details': {
                    'local_coherence': {
                        'score': float(local_quality),
                        'raw_value': float(avg_local),
                        'assessment': self.get_quality_label(local_quality)
                    },
                    'progression': {
                        'score': float(progression_quality),
                        'assessment': self.get_quality_label(progression_quality)
                    }
                }
            }
            
        except Exception as e:
            print(f"Error in analyze_concept_flow: {e}")
            return {
                'flow_score': 0.5,
                'quality': 'error',
                'details': {
                    'local_coherence': {'score': 0.5, 'assessment': 'error'},
                    'progression': {'score': 0.5, 'assessment': 'error'}
                }
            }
    
    def get_quality_label(self, score):
        """Get qualitative label for a score"""
        if score < 0.3:
            return "poor"
        elif score < 0.5:
            return "needs improvement"
        elif score < 0.7:
            return "adequate"
        elif score < 0.85:
            return "good"
        else:
            return "excellent"

def process_report(json_path):
    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract sections
    sections = data["sections"]
    
    # Create variables for complete report content
    original_report = " ".join([section_content.get("original", "") for section_content in sections.values()])
    improved_report = " ".join([section_content.get("improved", "") for section_content in sections.values()])
    
    # Calculate Flesch scores
    original_score = textstat.flesch_reading_ease(original_report)
    improved_score = textstat.flesch_reading_ease(improved_report)
    
    # Initialize coherence analyzer
    coherence_analyzer = ContextualCoherenceAnalyzer()
    
    # Add coherence metrics
    original_coherence = coherence_analyzer.analyze_contextual_coherence(original_report)
    improved_coherence = coherence_analyzer.analyze_contextual_coherence(improved_report)
    
    return original_report, improved_report, original_score, improved_score, original_coherence, improved_coherence

def count_technical_terms(text):
    """
    Count and calculate frequency of technical terms in text.
    Returns both raw counts and frequency normalized to 0-1 scale.
    """
    # Define technical terms related to semiconductors and general technical writing
    technical_terms = [
        # Semiconductor Materials & Properties
        'silicon', 'germanium', 'gallium', 'arsenide', 'substrate', 'wafer',
        'dopant', 'carrier', 'bandgap', 'conductivity', 'resistivity',
        'mobility', 'junction', 'interface', 'lattice',
        
        # Device Components & Types
        'transistor', 'mosfet', 'bjt', 'diode', 'capacitor', 'resistor',
        'gate', 'source', 'drain', 'channel', 'oxide', 'contact', 'interconnect',
        'cmos', 'analog', 'digital', 'integrated circuit', 'ic', 'chip',
        
        # Manufacturing & Processes
        'fabrication', 'lithography', 'etching', 'deposition', 'implantation',
        'oxidation', 'diffusion', 'annealing', 'metallization', 'planarization',
        'photoresist', 'masking', 'doping', 'packaging',
        
        # Electrical Parameters
        'voltage', 'current', 'threshold', 'leakage', 'power', 'frequency',
        'capacitance', 'resistance', 'impedance', 'noise', 'gain', 'efficiency',
        
        # Testing & Analysis
        'characterization', 'reliability', 'yield', 'defect', 'measurement',
        'simulation', 'modeling', 'analysis', 'verification', 'validation',
        
        # General Technical Terms
        'parameter', 'specification', 'methodology', 'optimization',
        'implementation', 'configuration', 'architecture', 'mechanism',
        'correlation', 'coefficient', 'algorithm', 'framework'
    ]
    
    # Get total word count
    total_words = len(text.split())
    
    # Count occurrences of each term
    term_counts = {}
    total_technical_terms = 0
    
    for term in technical_terms:
        # Use word boundaries to ensure we're matching whole words
        count = len(re.findall(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
        if count > 0:
            term_counts[term] = count
            total_technical_terms += count
    
    # Calculate frequencies
    if total_words > 0:
        # Overall technical term frequency (percentage of technical words)
        technical_frequency = total_technical_terms / total_words
        
        # Normalize to 0-1 scale
        # Assuming a good technical document might have 5-15% technical terms
        # Scale accordingly: anything above 15% will be capped at 1.0
        normalized_frequency = min(technical_frequency / 0.15, 1.0)
    else:
        technical_frequency = 0
        normalized_frequency = 0
    
    return {
        'raw_count': total_technical_terms,
        'unique_terms': len(term_counts),
        'term_frequencies': term_counts,
        'technical_frequency': technical_frequency,
        'normalized_score': normalized_frequency,
        'total_words': total_words
    }

def estimate_concept_hierarchy_depth(text):
    """
    Estimates the hierarchical depth of concepts in text using topic modeling
    and syntactic structure analysis, returning scores on a 0-1 scale.
    """
    # Initialize spaCy
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        nlp = spacy.load('en_core_web_sm')

    # Get topic hierarchy score (0-1)
    topic_score = analyze_topic_hierarchy_normalized(text)
    
    # Get sentence complexity score (0-1)
    syntax_score = analyze_sentence_complexity_normalized(text, nlp)
    
    # Combine scores with weights
    final_score = (topic_score * 0.6) + (syntax_score * 0.4)
    
    # Return detailed results for transparency
    return {
        'combined_score': final_score,
        'topic_hierarchy_score': topic_score,
        'syntax_complexity_score': syntax_score
    }

def analyze_topic_hierarchy_normalized(text, num_topics=5):
    """Uses LDA to identify topic hierarchy levels, with scores normalized to 0-1"""
    # Preprocess text
    sentences = text.split('.')
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english'
    )
    
    # Handle very short texts
    if len(sentences) < 3:
        return 0.2  # Very minimal topic structure
    
    try:
        doc_term_matrix = vectorizer.fit_transform(sentences)
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42
        )
        lda.fit(doc_term_matrix)
        
        # Analyze topic distribution
        topic_distributions = lda.transform(doc_term_matrix)
        
        # Calculate metrics
        # 1. Topic diversity: measure how evenly distributed the topics are
        topic_diversity = np.mean([np.std(dist) for dist in topic_distributions])
        normalized_diversity = min(topic_diversity / 0.3, 1.0)  # Normalize with sensible max
        
        # 2. Topic coherence: measure how distinct the topics are
        topic_words = []
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-10 - 1:-1]
            topic_words.append([feature_names[i] for i in top_features_ind])
        
        # Calculate overlap between topics (less overlap = better hierarchy)
        overlap_sum = 0
        topic_pairs = 0
        for i in range(len(topic_words)):
            for j in range(i+1, len(topic_words)):
                overlap = len(set(topic_words[i]) & set(topic_words[j]))
                overlap_sum += overlap
                topic_pairs += 1
        
        avg_overlap = overlap_sum / max(1, topic_pairs)
        # Normalize: 0 overlap → 1.0 score, 5+ words overlap → 0.0 score
        normalized_uniqueness = max(0, 1.0 - (avg_overlap / 5.0))
        
        # 3. Topic significance: measure how many significant topics there are
        significant_topics_count = sum(np.max(topic_distributions, axis=1) > 0.5)
        normalized_significance = min(significant_topics_count / 4.0, 1.0)  # Cap at 4 significant topics
        
        # Combined score with weights
        combined_score = (
            0.4 * normalized_diversity +
            0.3 * normalized_uniqueness +
            0.3 * normalized_significance
        )
        
        return combined_score
        
    except Exception as e:
        print(f"Error in topic analysis: {e}")
        return 0.3  # Default fallback score

def analyze_sentence_complexity_normalized(text, nlp):
    """Analyzes syntactic complexity through dependency parsing, normalized to 0-1"""
    try:
        doc = nlp(text)
        
        # Calculate metrics
        # 1. Tree depth: maximum dependency tree depth across sentences
        depths = []
        for sent in doc.sents:
            tree_depths = []
            for token in sent:
                depth = 1
                current = token
                while current.head != current:
                    depth += 1
                    current = current.head
                tree_depths.append(depth)
            
            if tree_depths:
                depths.append(max(tree_depths))
        
        if not depths:
            return 0.3  # Default for very short text
        
        avg_max_depth = np.mean(depths)
        # Normalize: optimal range for technical writing is 5-8 levels deep
        if avg_max_depth < 3:
            normalized_depth = 0.3  # Too simple
        elif avg_max_depth < 5:
            normalized_depth = 0.6  # Moderate complexity
        elif avg_max_depth <= 8:
            normalized_depth = 1.0  # Optimal complexity
        else:
            normalized_depth = 0.7  # Too complex
        
        # 2. Sentence structure variety
        # Calculate standard deviation of tree depths to measure variety
        depth_std = np.std(depths) if len(depths) > 1 else 0
        normalized_variety = min(depth_std / 2.0, 1.0)  # Normalize with sensible max
        
        # 3. Complex clause usage - count subordinate clauses
        clause_count = len([token for token in doc if token.dep_ in ('ccomp', 'xcomp', 'advcl')])
        normalized_clauses = min(clause_count / (len(list(doc.sents)) * 1.5), 1.0)  # Normalize per sentence
        
        # Combined score with weights
        combined_score = (
            0.5 * normalized_depth +
            0.3 * normalized_variety +
            0.2 * normalized_clauses
        )
        
        return combined_score
        
    except Exception as e:
        print(f"Error in syntax analysis: {e}")
        return 0.4  # Default fallback score

def count_examples(text):
    # Find phrases that indicate examples are being provided
    example_patterns = [
        r'for example[,:].*?[.!?]',  # "For example" followed by content
        r'for instance[,:].*?[.!?]',  # "For instance" followed by content
        r'such as.*?[.!?]',          # "Such as" followed by content
        r'e\.g\..*?[.!?]',           # "e.g." followed by content
        r'to illustrate[,:].*?[.!?]', # "To illustrate" followed by content
        r'specifically[,:].*?[.!?]',  # "Specifically" followed by content
        r'in particular[,:].*?[.!?]', # "In particular" followed by content
        r'namely[,:].*?[.!?]'         # "Namely" followed by content
    ]
    
    # Combine patterns and search for matches
    combined_pattern = '|'.join(example_patterns)
    examples = re.findall(combined_pattern, text, re.IGNORECASE)
    
    return len(examples)

def count_defined_terms(text):
    # Look for patterns where terms are defined
    definition_patterns = [
        r'\b\w+(?:\s+\w+){0,4}\s+is defined as\b.*?[.!?]',  # "X is defined as..."
        r'\b\w+(?:\s+\w+){0,4}\s+refers to\b.*?[.!?]',      # "X refers to..."
        r'\b\w+(?:\s+\w+){0,4}\s+means\b.*?[.!?]',          # "X means..."
        r'\b\w+(?:\s+\w+){0,4}\s+is\s+(?:a|an|the)\b.*?[.!?]', # "X is a/an/the..."
        r'\bdefin(?:e|ed|ing|ition of)\s+\w+(?:\s+\w+){0,4}\b.*?[.!?]', # "Define/definition of X..."
        r'\bin other words\b.*?[.!?]',                       # "In other words..."
        r'\bthat is\b.*?[.!?]',                             # "That is..."
        r'\bi\.e\.\b.*?[.!?]'                               # "i.e...."
    ]
    
    # Combine patterns and search for matches
    combined_pattern = '|'.join(definition_patterns)
    definitions = re.findall(combined_pattern, text, re.IGNORECASE)
    
    return len(definitions)

def save_results_to_json(results, output_path):
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the results to a JSON file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    
    print(f"Results saved to {output_path}")

def create_comparison_charts(results, output_dir):
    """
    Create bar charts comparing metrics between original and improved reports
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from results
    original = results["original_report"]
    improved = results["improved_report"]
    
    # Set up colors
    original_color = '#1f77b4'  # blue
    improved_color = '#2ca02c'  # green
    
    # Create figure 1: Basic metrics
    plt.figure(figsize=(12, 8))
    metrics = ['word_count', 'flesch_score']
    labels = ['Word Count', 'Flesch Reading Ease']
    x = np.arange(len(metrics))
    width = 0.35
    
    original_values = [original[m] for m in metrics]
    improved_values = [improved[m] for m in metrics]
    
    plt.bar(x - width/2, original_values, width, label='Original', color=original_color)
    plt.bar(x + width/2, improved_values, width, label='Improved', color=improved_color)
    
    plt.xlabel('Metrics')
    plt.xticks(x, labels)
    plt.title('Basic Readability Metrics Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(original_values):
        plt.text(i - width/2, v + max(original_values + improved_values)*0.02, 
                 str(v), ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(improved_values):
        plt.text(i + width/2, v + max(original_values + improved_values)*0.02, 
                 str(v), ha='center', va='bottom', fontweight='bold')
    
    # Save figure
    chart_path = os.path.join(output_dir, f"basic_metrics_{results['timestamp']}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Created basic metrics chart: {chart_path}")
    plt.close()
    
    # Create figure 2: Technical metrics
    plt.figure(figsize=(14, 8))
    tech_metrics = [
        'concept_hierarchy_depth',
        'actionable_recommendations_count', 
        'technical_term_count',
        'example_count', 
        'defined_terms_count'
    ]
    
    tech_labels = [
        'Concept Hierarchy\nDepth',
        'Actionable\nRecommendations', 
        'Technical\nTerms',
        'Examples', 
        'Defined\nTerms'
    ]
    
    x = np.arange(len(tech_metrics))
    
    original_tech_values = [original[m] for m in tech_metrics]
    improved_tech_values = [improved[m] for m in tech_metrics]
    
    plt.bar(x - width/2, original_tech_values, width, label='Original', color=original_color)
    plt.bar(x + width/2, improved_tech_values, width, label='Improved', color=improved_color)
    
    plt.xlabel('Technical Metrics')
    plt.xticks(x, tech_labels)
    plt.title('Technical and Clarity Metrics Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(original_tech_values):
        plt.text(i - width/2, v + max(original_tech_values + improved_tech_values)*0.02,
                 str(v), ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(improved_tech_values):
        plt.text(i + width/2, v + max(original_tech_values + improved_tech_values)*0.02,
                 str(v), ha='center', va='bottom', fontweight='bold')
    
    # Save figure
    chart_path = os.path.join(output_dir, f"technical_metrics_{results['timestamp']}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Created technical metrics chart: {chart_path}")
    plt.close()
    
    # Create figure 4: Percentage changes
    plt.figure(figsize=(12, 8))
    
    # Calculate percentage changes for all metrics
    percent_changes = []
    metric_names = []
    
    # Basic metrics
    word_count_change = results["comparison"]["word_count_percent_change"]
    flesch_score_change = (results["comparison"]["flesch_score_difference"] / original["flesch_score"]) * 100 if original["flesch_score"] != 0 else 0
    
    # Technical metrics
    concept_depth_change = results["comparison"]["concept_depth_difference"]
    recommendations_change = results["comparison"]["recommendations_difference"]
    tech_terms_change = results["comparison"]["technical_terms_difference"]
    examples_change = results["comparison"]["examples_difference"]
    defined_terms_change = results["comparison"]["defined_terms_difference"]
    
    # Add all changes and labels
    percent_changes = [
        word_count_change, 
        flesch_score_change, 
        concept_depth_change,
        recommendations_change, 
        tech_terms_change,
        examples_change, 
        defined_terms_change
    ]
    
    metric_names = [
        'Word Count', 
        'Flesch Score', 
        'Concept Depth',
        'Recommendations', 
        'Technical Terms',
        'Examples', 
        'Defined Terms'
    ]
    
    # Create color map based on whether change is positive or negative
    colors = ['#2ca02c' if change >= 0 else '#d62728' for change in percent_changes]
    
    # Sort by absolute percentage change (optional)
    sorted_indices = np.argsort(np.abs(percent_changes))[::-1]  # Descending order
    percent_changes = [percent_changes[i] for i in sorted_indices]
    metric_names = [metric_names[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(metric_names, percent_changes, color=colors)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Percentage Change (%)')
    plt.title('Percentage Change in Metrics (Improved vs. Original)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1 if width >= 0 else width - 5
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontweight='bold')
    
    # Save figure
    chart_path = os.path.join(output_dir, f"percentage_changes_{results['timestamp']}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Created percentage changes chart: {chart_path}")
    plt.close()
    
    return True

def llm_evaluate_report(original_report, improved_report, api_key):
    """
    Use an LLM (OpenAI) to evaluate the original and improved reports
    Returns the evaluation results with scores on a 0-1 scale
    """
    # Create a combined prompt to evaluate both texts
    prompt = f"""I'll provide you with two versions of a report: an original version and an improved version.
    Please evaluate both versions for technical depth, clarity, and overall effectiveness.
    
    Provide a score from 0.0-1.0 for each category, with justification and specific observations.
    
    ORIGINAL REPORT:
    ```
    {original_report[:6000]}  # Limiting to first 6000 chars to keep within context window
    ```
    
    IMPROVED REPORT:
    ```
    {improved_report[:6000]}  # Limiting to first 6000 chars to keep within context window
    ```
    
    Format your response as a JSON object with this exact structure:
    {{
        "original": {{
            "technical_depth": {{
                "score": <0.0-1.0>,
                "justification": "detailed explanation"
            }},
            "clarity": {{
                "score": <0.0-1.0>,
                "justification": "detailed explanation"
            }},
            "overall": {{
                "score": <0.0-1.0>,
                "justification": "detailed explanation"
            }}
        }},
        "improved": {{
            "technical_depth": {{
                "score": <0.0-1.0>,
                "justification": "detailed explanation"
            }},
            "clarity": {{
                "score": <0.0-1.0>,
                "justification": "detailed explanation"
            }},
            "overall": {{
                "score": <0.0-1.0>,
                "justification": "detailed explanation"
            }}
        }},
        "comparison": {{
            "technical_depth_difference": <improved_score - original_score>,
            "clarity_difference": <improved_score - original_score>,
            "overall_difference": <improved_score - original_score>,
            "summary": "Brief summary of the main improvements or changes"
        }}
    }}
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        # Extract the JSON response from OpenAI's message
        response_text = response.choices[0].message.content
        
        # Find the JSON response in the text (in case GPT adds additional commentary)
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            evaluation = json.loads(json_str)
        else:
            evaluation = json.loads(response_text)
            
        return evaluation

    except Exception as e:
        print(f"Error in LLM evaluation: {str(e)}")
        # Return default values in case of error
        return {
            'original': {
                'technical_depth': {'score': 0.5, 'justification': "Error in evaluation process"},
                'clarity': {'score': 0.5, 'justification': "Error in evaluation process"},
                'overall': {'score': 0.5, 'justification': "Error in evaluation process"}
            },
            'improved': {
                'technical_depth': {'score': 0.5, 'justification': "Error in evaluation process"},
                'clarity': {'score': 0.5, 'justification': "Error in evaluation process"},
                'overall': {'score': 0.5, 'justification': "Error in evaluation process"}
            },
            'comparison': {
                'technical_depth_difference': 0.0,
                'clarity_difference': 0.0,
                'overall_difference': 0.0,
                'summary': "Error in evaluation process"
            }
        }

def create_llm_comparison_chart(llm_results, output_dir, timestamp):
    """
    Create a bar chart comparing LLM evaluation scores (0-1 scale)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from results
    original = llm_results["original"]
    improved = llm_results["improved"]
    
    # Set up colors
    original_color = '#1f77b4'  # blue
    improved_color = '#2ca02c'  # green
    
    # Create figure: LLM evaluation scores
    plt.figure(figsize=(12, 8))
    metrics = ['technical_depth', 'clarity', 'overall']
    labels = ['Technical Depth', 'Clarity', 'Overall']
    x = np.arange(len(metrics))
    width = 0.35
    
    original_values = [original[m]['score'] for m in metrics]
    improved_values = [improved[m]['score'] for m in metrics]
    
    plt.bar(x - width/2, original_values, width, label='Original', color=original_color)
    plt.bar(x + width/2, improved_values, width, label='Improved', color=improved_color)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score (0-1)')
    plt.xticks(x, labels)
    plt.title('LLM Evaluation Scores Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(original_values):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(improved_values):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add difference arrows and labels
    for i in range(len(metrics)):
        diff = improved_values[i] - original_values[i]
        if diff != 0:
            # Arrow color based on difference
            arrow_color = 'green' if diff > 0 else 'red'
            
            # Position arrow between the bars
            arrow_x = i
            arrow_y_start = min(original_values[i], improved_values[i]) + (abs(diff) / 2)
            
            # Draw the arrow
            plt.annotate(
                f"{diff:+.3f}",
                xy=(arrow_x, arrow_y_start),
                xytext=(arrow_x, arrow_y_start - 0.02 if diff < 0 else arrow_y_start + 0.02),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2),
                ha='center',
                va='center',
                fontweight='bold',
                color=arrow_color
            )
    
    # Set y-axis limit to 0-1 with some padding
    plt.ylim(0, 1.05)
    
    # Save figure
    chart_path = os.path.join(output_dir, f"llm_evaluation_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Created LLM evaluation chart: {chart_path}")
    plt.close()
    
    return chart_path

def create_coherence_chart(results, output_dir, timestamp):
    """Create a chart comparing coherence metrics"""
    plt.figure(figsize=(12, 8))
    
    # Update metrics to only include flow score
    metrics = ['Flow Score']
    original_values = [
        results['original_report']['contextual_coherence']['concept_flow']['flow_score']
    ]
    improved_values = [
        results['improved_report']['contextual_coherence']['concept_flow']['flow_score']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original', color='#1f77b4')
    plt.bar(x + width/2, improved_values, width, label='Improved', color='#2ca02c')
    
    plt.xlabel('Coherence Metrics')
    plt.ylabel('Score')
    plt.title('Contextual Coherence Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    chart_path = os.path.join(output_dir, f"coherence_metrics_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

def calculate_technical_depth(text):
    """
    Calculate technical depth metrics using frequency-based technical term analysis and LLM evaluation
    """
    # Get technical metrics with frequency
    tech_metrics = count_technical_terms(text)
    normalized_term_score = tech_metrics['normalized_score']
    
    # Get concept hierarchy depth
    concept_hierarchy_depth = estimate_concept_hierarchy_depth(text)
    # Fix: use the combined_score directly as it's already normalized to 0-1
    normalized_depth = concept_hierarchy_depth['combined_score']
    
    # Initialize result dictionary
    result = {
        'technical_term_metrics': tech_metrics,
        'concept_hierarchy_depth': normalized_depth
    }
    
    # Add LLM evaluation
    try:
        prompt = f"""Evaluate the technical depth of the following text. Consider:
        1. Complexity and sophistication of technical concepts
        2. Depth of technical explanations
        3. Use of domain-specific terminology
        4. Technical accuracy and precision
        
        Provide a score from 0.0-1.0 and a brief justification.
        
        Text to evaluate:
        ```
        {text[:6000]}  # Limiting to first 6000 chars to keep within context window
        ```
        
        Format your response as a JSON object with this structure:
        {{
            "score": <0.0-1.0>,
            "justification": "brief explanation"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        # Extract the JSON response
        import re
        import json
        json_match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content)
        if json_match:
            llm_evaluation = json.loads(json_match.group(0))
            result['llm_evaluation'] = {
                'score': float(llm_evaluation['score']),
                'justification': llm_evaluation['justification']
            }
        
    except Exception as e:
        print(f"Warning: LLM evaluation failed: {str(e)}")
        result['llm_evaluation'] = {
            'score': 0.5,  # Default middle score
            'justification': "LLM evaluation failed"
        }
    
    # Calculate combined score with LLM evaluation
    result['combined_score'] = (
        0.3 * normalized_term_score +    # 30% weight to term frequency
        0.3 * normalized_depth +         # 30% weight to concept hierarchy
        0.4 * result['llm_evaluation']['score']  # 40% weight to LLM evaluation
    )
    
    return result

def calculate_clarity(text):
    """
    Calculate clarity and understandability metrics for a given text.
    
    Returns:
        dict: Dictionary containing clarity metrics (all normalized to 0-1):
            - flesch_score: Normalized Flesch reading ease score
            - defined_terms_count: Normalized number of defined terms
            - example_count: Normalized number of examples
            - llm_evaluation: LLM-based clarity evaluation
    """
    # Calculate basic metrics
    flesch_score = textstat.flesch_reading_ease(text)
    
    # Get technical term metrics to use as a base for determining how many terms should be defined
    tech_metrics = count_technical_terms(text)
    unique_technical_terms = tech_metrics['unique_terms']
    
    # Count defined terms and examples
    defined_terms_count = count_defined_terms(text)
    example_count = count_examples(text)
    
    # Calculate total sentences to estimate concepts that could benefit from examples
    sentences = text.split('.')
    total_sentences = len([s for s in sentences if len(s.strip()) > 10])  # Only count meaningful sentences
    estimated_concepts = max(1, total_sentences // 3)  # Estimate one concept per three sentences
    
    # Normalize Flesch score for technical content (target ~30)
    if flesch_score <= 10:  # Extremely complex, even for technical content
        normalized_flesch = 0.2
    elif flesch_score <= 20:  # Very complex technical content
        normalized_flesch = 0.4
    elif flesch_score <= 35:  # Optimal range for technical content
        normalized_flesch = 1.0
    elif flesch_score <= 50:  # Slightly more readable than needed
        normalized_flesch = 0.8
    else:  # Too simple for technical audience
        normalized_flesch = 0.6
    
    # Normalize definitions relative to unique technical terms that should be defined
    # Optimal is to define 50-80% of technical terms
    definition_ratio = defined_terms_count / max(1, unique_technical_terms) if unique_technical_terms > 0 else 0
    if definition_ratio > 0.8:  # More definitions than needed
        normalized_defined = 0.8
    elif definition_ratio >= 0.5:  # Optimal range
        normalized_defined = 1.0
    elif definition_ratio >= 0.3:  # Acceptable but could use more
        normalized_defined = 0.7
    elif definition_ratio > 0:  # Too few definitions
        normalized_defined = 0.4 * (definition_ratio / 0.3)  # Scale from 0 to 0.4
    else:  # No definitions
        normalized_defined = 0
    
    # Normalize examples relative to estimated concepts that benefit from examples
    # Optimal is to provide examples for 30-50% of key concepts
    example_ratio = example_count / max(1, estimated_concepts) if estimated_concepts > 0 else 0
    if example_ratio > 0.6:  # More examples than needed
        normalized_examples = 0.9
    elif example_ratio >= 0.3:  # Optimal range
        normalized_examples = 1.0
    elif example_ratio >= 0.1:  # Acceptable but could use more
        normalized_examples = 0.6
    elif example_ratio > 0:  # Too few examples
        normalized_examples = 0.3 * (example_ratio / 0.1)  # Scale from 0 to 0.3
    else:  # No examples
        normalized_examples = 0
    
    # Add LLM evaluation
    try:        
        prompt = f"""Evaluate the clarity and understandability of the following text. Consider:
        1. Clear and concise explanations
        2. Logical flow of ideas
        3. Effective use of examples and definitions
        4. Accessibility to the target audience
        
        Provide a score from 0.0-1.0 and a brief justification.
        
        Text to evaluate:
        ```
        {text[:6000]}  # Limiting to first 6000 chars to keep within context window
        ```
        
        Format your response as a JSON object with this structure:
        {{
            "score": <0.0-1.0>,
            "justification": "brief explanation"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        # Extract the JSON response
        json_match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content)
        if json_match:
            llm_evaluation = json.loads(json_match.group(0))
            llm_score = float(llm_evaluation['score'])
            llm_justification = llm_evaluation['justification']
        else:
            llm_score = 0.5
            llm_justification = "Failed to parse LLM response"
            
    except Exception as e:
        print(f"Warning: LLM evaluation failed: {str(e)}")
        llm_score = 0.5
        llm_justification = "LLM evaluation failed"
    
    # Calculate combined score with weights
    combined_score = (
        0.25 * normalized_flesch +     # 25% weight to readability
        0.15 * normalized_defined +    # 15% weight to defined terms
        0.15 * normalized_examples +   # 15% weight to examples
        0.45 * llm_score               # 45% weight to LLM evaluation
    )
    
    return {
        'flesch_score': normalized_flesch,
        'defined_terms_count': normalized_defined,
        'example_count': normalized_examples,
        'definition_ratio': definition_ratio,  # Added for transparency
        'example_ratio': example_ratio,        # Added for transparency
        'llm_evaluation': {
            'score': llm_score,
            'justification': llm_justification
        },
        'combined_score': combined_score
    }

def calculate_structure(text):
    """
    Calculate structure metrics for a given text.
    
    Returns:
        dict: Dictionary containing structure metrics:
            - coherence: Contextual coherence metrics (normalized to 0-1)
            - llm_evaluation: LLM-based structure evaluation
    """
    # Get coherence metrics
    coherence_analyzer = ContextualCoherenceAnalyzer()
    coherence = coherence_analyzer.analyze_contextual_coherence(text)
    
    # Ensure we have a valid flow score
    flow_score = coherence.get('concept_flow', {}).get('flow_score', 0.5)
    if np.isnan(flow_score):
        flow_score = 0.5  # Default to neutral score if NaN
    
    # Add LLM evaluation
    try:
        prompt = f"""Evaluate the structural organization of the following text. Consider:
        1. Logical organization and flow
        2. Effective use of paragraphs and sections
        3. Transitions between ideas
        4. Overall document structure
        
        Provide a score from 0.0-1.0 and a brief justification.
        
        Text to evaluate:
        ```
        {text[:6000]}  # Limiting to first 6000 chars to keep within context window
        ```
        
        Format your response as a JSON object with this structure:
        {{
            "score": <0.0-1.0>,
            "justification": "brief explanation"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        # Extract the JSON response
        json_match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content)
        if json_match:
            llm_evaluation = json.loads(json_match.group(0))
            llm_score = float(llm_evaluation['score'])
            llm_justification = llm_evaluation['justification']
        else:
            llm_score = 0.5
            llm_justification = "Failed to parse LLM response"
            
    except Exception as e:
        print(f"Warning: LLM evaluation failed: {str(e)}")
        llm_score = 0.5
        llm_justification = "LLM evaluation failed"
    
    # Calculate combined score with weights and ensure no NaN
    try:
        combined_score = (
            0.6 * flow_score +  # 60% weight to automated coherence
            0.4 * llm_score    # 40% weight to LLM evaluation
        )
        if np.isnan(combined_score):
            combined_score = 0.5  # Default to neutral score if calculation fails
    except Exception as e:
        print(f"Warning: Error calculating combined structure score: {str(e)}")
        combined_score = 0.5
    
    return {
        'coherence': coherence,
        'llm_evaluation': {
            'score': llm_score,
            'justification': llm_justification
        },
        'combined_score': float(combined_score)  # Ensure we return a float, not numpy float
    }

def calculate_final_weighted_score(text, llm_technical_depth=None, llm_clarity=None):
    """
    Calculate a comprehensive weighted score for a text combining all metrics.
    
    Args:
        text (str): The text to analyze
        llm_technical_depth (dict, optional): LLM evaluation for technical depth
        llm_clarity (dict, optional): LLM evaluation for clarity
    
    Returns:
        dict: Dictionary containing all metrics and the weighted score
    """
    # Calculate all component metrics
    technical_metrics = calculate_technical_depth(text)
    clarity_metrics = calculate_clarity(text)
    structure_metrics = calculate_structure(text)
    
    # Create consolidated metrics dictionary
    metrics = {
        'technical_term_count': technical_metrics['technical_term_metrics']['raw_count'],
        'concept_hierarchy_depth': technical_metrics['concept_hierarchy_depth'],
        'flesch_score': clarity_metrics['flesch_score'],
        'defined_terms_count': clarity_metrics['defined_terms_count'],
        'example_count': clarity_metrics['example_count'],
        'contextual_coherence': structure_metrics['coherence'],
        'word_count': len(text.split())
    }
    
    # Add LLM scores if available
    llm_results = {
        'technical_depth': {'score': 0.5},  # Default values
        'clarity': {'score': 0.5}
    }
    
    if llm_technical_depth is not None:
        llm_results['technical_depth'] = llm_technical_depth
    
    if llm_clarity is not None:
        llm_results['clarity'] = llm_clarity
    
    # Calculate weighted score
    weighted_scores = calculate_weighted_score(metrics, llm_results)
    
    # Return combined results
    return {
        'metrics': metrics,
        'weighted_score': weighted_scores
    }

def calculate_weighted_score(report_metrics, llm_results):
    """
    Calculate a weighted score (0-1) based on various metrics
    
    Weights are grouped into three main categories:
    1. Technical Depth (45% total)
    2. Clarity & Understandability (35%)
    3. Structure (20%)
    """
    
    # Technical Depth (45% total)
    technical_weights = {
        'technical_term_count': 0.15,  # 15%
        'concept_hierarchy_depth': 0.10,  # 10%
        'llm_technical_depth': 0.20,  # 20%
    }
    
    # Clarity & Understandability (35% total)
    clarity_weights = {
        'flesch_score': 0.10,  # 10%
        'defined_terms_count': 0.05,  # 5%
        'example_count': 0.05,  # 5%
        'llm_clarity': 0.15,  # 15%
    }
    
    # Structure (20% total)
    structure_weights = {
        'coherence_flow_score': 0.20,  # 20%
    }

    # Calculate normalized scores (0-1 scale)
    scores = {
        # Technical Depth
        'technical_term_count': min(report_metrics['technical_term_count'] * 0.05, 1.0),
        'concept_hierarchy_depth': report_metrics['concept_hierarchy_depth'] / 5.0,  # Already 1-5 scale
        'llm_technical_depth': llm_results['technical_depth']['score'],
        
        # Clarity & Understandability
        'flesch_score': min(max(report_metrics['flesch_score'] / 100.0, 0), 1.0),
        'defined_terms_count': min(report_metrics['defined_terms_count'] * 0.1, 1.0),
        'example_count': min(report_metrics['example_count'] * 0.2, 1.0),
        'llm_clarity': llm_results['clarity']['score'],
        
        # Structure
        'coherence_flow_score': report_metrics['contextual_coherence']['concept_flow']['flow_score']  # Already 0-1
    }
    
    # Calculate weighted scores
    technical_score = sum(scores[metric] * weight for metric, weight in technical_weights.items())
    clarity_score = sum(scores[metric] * weight for metric, weight in clarity_weights.items())
    structure_score = sum(scores[metric] * weight for metric, weight in structure_weights.items())
    
    # Calculate final score
    final_score = technical_score + clarity_score + structure_score
    
    return {
        'final_score': round(final_score, 3),
        'component_scores': {
            'technical_depth': round(technical_score, 3),
            'clarity': round(clarity_score, 3),
            'structure': round(structure_score, 3)
        },
        'detailed_scores': {metric: round(score, 3) for metric, score in scores.items()}
    }

def compare_report_scores(original_metrics, improved_metrics, llm_results):
    """
    Compare the weighted scores between original and improved reports
    """
    original_score = calculate_weighted_score(original_metrics, llm_results['original'])
    improved_score = calculate_weighted_score(improved_metrics, llm_results['improved'])
    
    score_difference = improved_score['final_score'] - original_score['final_score']
    percent_improvement = (score_difference / original_score['final_score']) * 100
    
    return {
        'original': original_score,
        'improved': improved_score,
        'difference': round(score_difference, 2),
        'percent_improvement': round(percent_improvement, 2),
        'component_differences': {
            'technical_depth': round(improved_score['component_scores']['technical_depth'] - 
                                   original_score['component_scores']['technical_depth'], 2),
            'clarity': round(improved_score['component_scores']['clarity'] - 
                           original_score['component_scores']['clarity'], 2),
            'structure': round(improved_score['component_scores']['structure'] - 
                             original_score['component_scores']['structure'], 2)
        }
    }

def create_weighted_scores_chart(weighted_scores, output_dir, timestamp):
    """
    Create a detailed chart showing the weighted scores comparison
    """
    plt.figure(figsize=(15, 10))
    
    # Create data for the plot
    categories = ['Technical Depth', 'Clarity', 'Structure', 'Final Score']
    original_values = [
        weighted_scores['original']['component_scores']['technical_depth'],
        weighted_scores['original']['component_scores']['clarity'],
        weighted_scores['original']['component_scores']['structure'],
        weighted_scores['original']['final_score']
    ]
    improved_values = [
        weighted_scores['improved']['component_scores']['technical_depth'],
        weighted_scores['improved']['component_scores']['clarity'],
        weighted_scores['improved']['component_scores']['structure'],
        weighted_scores['improved']['final_score']
    ]
    
    # Set up the bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, original_values, width, label='Original', color='#1f77b4', alpha=0.8)
    plt.bar(x + width/2, improved_values, width, label='Improved', color='#2ca02c', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Score Components')
    plt.ylabel('Score (0-1)')
    plt.title('Weighted Score Comparison')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(original_values):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(improved_values):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement arrows and labels
    for i in range(len(categories)):
        diff = improved_values[i] - original_values[i]
        if diff != 0:
            # Arrow color based on difference
            arrow_color = 'green' if diff > 0 else 'red'
            
            # Position arrow between the bars
            arrow_x = i
            arrow_y_start = min(original_values[i], improved_values[i]) + (abs(diff) / 2)
            
            # Draw the arrow
            plt.annotate(
                f"{diff:+.3f}",
                xy=(arrow_x, arrow_y_start),
                xytext=(arrow_x, arrow_y_start - 0.02 if diff < 0 else arrow_y_start + 0.02),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2),
                ha='center',
                va='center',
                fontweight='bold',
                color=arrow_color
            )
    
    # Add percentage improvement as a subtitle
    plt.figtext(
        0.5, 0.02,
        f"Overall Improvement: {weighted_scores['percent_improvement']:+.3f}%",
        ha='center',
        fontsize=10,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    
    # Set y-axis limit to 0-1 with some padding
    plt.ylim(0, 1.05)
    
    # Save the chart
    chart_path = os.path.join(output_dir, f"weighted_scores_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Created weighted scores chart: {chart_path}")
    plt.close()
    
    return chart_path

def evaluate_citation_accuracy(text: str, referenced_papers: Dict) -> Dict:
    """
    Evaluate factual accuracy by comparing content against reference data.
    
    Args:
        text (str): The text to evaluate for factual accuracy
        referenced_papers (Dict): Dictionary of referenced papers with their content
        
    Returns:
        Dict containing:
            - score (float): Overall factual accuracy score (0-1)
            - citation_analysis (list): Analysis of factual accuracy
            - needs_improvement (bool): Whether the content needs factual improvement
            - improvement_suggestions (dict): Specific suggestions for improving accuracy
    """
    # Prepare reference content
    reference_content = ""
    for paper_id, paper_info in referenced_papers.items():
        reference_content += f"[{paper_info['citation_id']}] {paper_info.get('title', 'Untitled')}\n"
        reference_content += f"Abstract: {paper_info.get('abstract', '')}\n"
        if paper_info.get('chunks'):
            reference_content += "Key content:\n"
            for chunk in paper_info.get('chunks', []):
                reference_content += f"- {chunk}\n"
        reference_content += "\n---\n\n"
    
    # Create prompt for factual accuracy evaluation
    prompt = f"""Evaluate the factual accuracy of the text by comparing it against the provided reference data.

TEXT TO EVALUATE:
```
{text[:3000]}  # First 3000 chars of text
```

REFERENCE DATA:
```
{reference_content[:4000]}  # First 4000 chars of reference content
```

Please analyze the text's factual accuracy based on the reference data. Consider:
1. Whether claims in the text are supported by the reference data
2. If there are any factual errors or misrepresentations
3. How well the text reflects the information from the references

Format your response as a JSON object with this structure:
{{
    "score": <0.0-1.0>,  # Overall factual accuracy score
    "analysis": [
        {{
            "claim": "specific claim or statement from text",
            "accuracy": <0.0-1.0>,  # Accuracy score for this claim
            "reference_support": "relevant information from references or 'Not supported'",
            "explanation": "brief explanation of accuracy rating"
        }},
        # Additional claims...
    ],
    "needs_improvement": <true/false>,
    "improvement_suggestions": "specific suggestions for improving factual accuracy"
}}

IMPORTANT: Keep all claims and explanations short and focused.
Assign a SINGLE score that reflects the OVERALL factual accuracy.
Analyze at most 3-4 key claims from the text.

Scoring criteria:
- 1.0: All claims are fully supported by the references
- 0.7-0.9: Most claims are accurate with minor discrepancies
- 0.4-0.6: Some claims are accurate but others lack support
- 0.1-0.3: Many claims lack support or contradict references
- 0.0: No claims are supported by the references"""

    try:
        # Get LLM evaluation
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1
        )
        
        # Parse response
        json_match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content)
        if json_match:
            evaluation = json.loads(json_match.group(0))
            score = float(evaluation.get("score", 0.5))
            analysis = evaluation.get("analysis", [])
            needs_improvement = evaluation.get("needs_improvement", score < 0.7)
            improvement_suggestions = evaluation.get("improvement_suggestions", "")
        else:
            score = 0.5
            analysis = []
            needs_improvement = True
            improvement_suggestions = "Failed to parse LLM response"
        
        # Transform analysis into citation_analysis format for compatibility
        citation_analysis = []
        for i, claim_analysis in enumerate(analysis):
            citation_analysis.append({
                "citation_id": f"claim_{i+1}",  # Generate unique IDs for each claim
                "score": float(claim_analysis.get("accuracy", 0.5)),
                "justification": claim_analysis.get("explanation", ""),
                "contexts": [claim_analysis.get("claim", "")]
            })
        
        # If no specific claims were analyzed, add a general analysis
        if not citation_analysis:
            citation_analysis.append({
                "citation_id": "overall",
                "score": score,
                "justification": "General accuracy assessment",
                "contexts": ["Overall text content"]
            })
        
        # Format improvement suggestions
        improvement_suggestions_dict = {}
        if needs_improvement:
            improvement_suggestions_dict["overall"] = {
                "current_score": score,
                "contexts": ["Overall text content"],
                "suggestion": improvement_suggestions
            }
        
        return {
            "score": score,
            "citation_analysis": citation_analysis,
            "needs_improvement": needs_improvement,
            "improvement_suggestions": improvement_suggestions_dict
        }
            
    except Exception as e:
        print(f"Error evaluating factual accuracy: {str(e)}")
        return {
            "score": 0.5,
            "citation_analysis": [{
                "citation_id": "overall",
                "score": 0.5,
                "justification": f"Error evaluating factual accuracy: {str(e)}",
                "contexts": ["Error occurred during evaluation"]
            }],
            "needs_improvement": True,
            "improvement_suggestions": {"overall": {
                "current_score": 0.5,
                "contexts": ["Error occurred"],
                "suggestion": "Manual review recommended due to evaluation error"
            }}
        }

if __name__ == "__main__":
    # Create base output directory
    base_output_dir = "evaluation_results"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize lists to store overall scores
    original_overall_scores = []
    improved_overall_scores = []
    
    # Get all JSON files from rewritten_reports directory
    rewritten_reports_dir = "rewritten_reports"
    json_files = [f for f in os.listdir(rewritten_reports_dir) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} reports to process")
    
    # Process each report
    for json_file in json_files:
        report_name = json_file.replace('.json', '')
        json_path = os.path.join(rewritten_reports_dir, json_file)
            
        print(f"\nProcessing: {report_name}")
        
        # Create report-specific output directory
        report_output_dir = os.path.join(base_output_dir, report_name)
        charts_dir = os.path.join(report_output_dir, "charts")
        os.makedirs(report_output_dir, exist_ok=True)
        os.makedirs(charts_dir, exist_ok=True)
        
        try:
            # Load and process the report
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            sections = data["sections"]
            
            # Process the report
            original_report, improved_report, original_score, improved_score, original_coherence, improved_coherence = process_report(json_path)
            
            # Calculate report lengths
            original_length = len(original_report.split())
            improved_length = len(improved_report.split())
            
            # Calculate technical metrics for original report
            original_concept_depth = estimate_concept_hierarchy_depth(original_report)
            original_technical_terms = count_technical_terms(original_report)
            original_examples = count_examples(original_report)
            original_defined_terms = count_defined_terms(original_report)
            
            # Calculate technical metrics for improved report
            improved_concept_depth = estimate_concept_hierarchy_depth(improved_report)
            improved_technical_terms = count_technical_terms(improved_report)
            improved_examples = count_examples(improved_report)
            improved_defined_terms = count_defined_terms(improved_report)
            
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get LLM evaluation
            print("\nRequesting LLM evaluation...")
            llm_results = llm_evaluate_report(original_report, improved_report, "not-used")
            
            # Create a results dictionary including LLM evaluation
            results = {
                "file_analyzed": json_path,
                "timestamp": timestamp,
                "original_report": {
                    "word_count": original_length,
                    "flesch_score": round(original_score, 2),
                    "concept_hierarchy_depth": original_concept_depth['combined_score'],
                    "actionable_recommendations_count": 0,
                    "technical_term_count": original_technical_terms,
                    "example_count": original_examples,
                    "defined_terms_count": original_defined_terms,
                    "contextual_coherence": {
                        "concept_flow": original_coherence['concept_flow'],
                    }
                },
                "improved_report": {
                    "word_count": improved_length,
                    "flesch_score": round(improved_score, 2),
                    "concept_hierarchy_depth": improved_concept_depth['combined_score'],
                    "actionable_recommendations_count": 0,
                    "technical_term_count": improved_technical_terms,
                    "example_count": improved_examples,
                    "defined_terms_count": improved_defined_terms,
                    "contextual_coherence": {
                        "concept_flow": improved_coherence['concept_flow'],
                    }
                },
                "comparison": {
                    "word_count_difference": improved_length - original_length,
                    "word_count_percent_change": round(((improved_length - original_length) / original_length) * 100, 2),
                    "flesch_score_difference": round(improved_score - original_score, 2),
                    "concept_depth_difference": improved_concept_depth['combined_score'] - original_concept_depth['combined_score'],
                    "technical_terms_difference": improved_technical_terms - original_technical_terms,
                    "examples_difference": improved_examples - original_examples,
                    "defined_terms_difference": improved_defined_terms - original_defined_terms,
                    "coherence_differences": {
                        "flow_score_change": improved_coherence['concept_flow']['flow_score'] - 
                                           original_coherence['concept_flow']['flow_score']
                    }
                },
                "llm_evaluation": llm_results
            }
            
            # Print the simple report
            print("REPORT STATISTICS")
            print("=================")
            print(f"Original report word count: {original_length}")
            print(f"Improved report word count: {improved_length}")
            print(f"Original Flesch score: {original_score:.2f}")
            print(f"Improved Flesch score: {improved_score:.2f}")
            
            # Show change
            diff = improved_score - original_score
            direction = "higher" if diff > 0 else "lower" if diff < 0 else "unchanged"
            print(f"Readability change: {abs(diff):.2f} points {direction}")
            
            # Print technical metrics
            print("\nTECHNICAL METRICS")
            print("=================")
            print(f"Concept hierarchy depth: {original_concept_depth['combined_score']:.2f} → {improved_concept_depth['combined_score']:.2f}")
            print(f"Technical terms: {original_technical_terms} → {improved_technical_terms}")
            print(f"Examples: {original_examples} → {improved_examples}")
            print(f"Defined terms: {original_defined_terms} → {improved_defined_terms}")
            
            # Print LLM evaluation
            print("\nLLM EVALUATION")
            print("==============")
            print("Original Report:")
            print(f"Technical Depth: {llm_results['original']['technical_depth']['score']}")
            print(f"Clarity: {llm_results['original']['clarity']['score']}")
            print(f"Overall: {llm_results['original']['overall']['score']}")
            
            print("\nImproved Report:")
            print(f"Technical Depth: {llm_results['improved']['technical_depth']['score']}")
            print(f"Clarity: {llm_results['improved']['clarity']['score']}")
            print(f"Overall: {llm_results['improved']['overall']['score']}")
            
            print("\nComparison:")
            print(f"Technical Depth Difference: {llm_results['comparison']['technical_depth_difference']}")
            print(f"Clarity Difference: {llm_results['comparison']['clarity_difference']}")
            print(f"Overall Difference: {llm_results['comparison']['overall_difference']}")
            print(f"Summary: {llm_results['comparison']['summary']}")
            
            # Calculate weighted scores
            weighted_scores = compare_report_scores(
                results['original_report'],
                results['improved_report'],
                results['llm_evaluation']
            )

            # Add scores to results
            results['weighted_scores'] = weighted_scores

            # Print weighted score comparison
            print("\nWEIGHTED SCORES")
            print("===============")
            print(f"Original Report: {weighted_scores['original']['final_score']}")
            print(f"Improved Report: {weighted_scores['improved']['final_score']}")
            print(f"Improvement: {weighted_scores['difference']} points ({weighted_scores['percent_improvement']}%)")

            print("\nComponent Improvements:")
            print(f"Technical Depth: {weighted_scores['component_differences']['technical_depth']}")
            print(f"Clarity: {weighted_scores['component_differences']['clarity']}")
            print(f"Structure: {weighted_scores['component_differences']['structure']}")
            
            # Store the overall scores
            original_overall_scores.append(weighted_scores['original']['final_score'])
            improved_overall_scores.append(weighted_scores['improved']['final_score'])
            
            # Update output paths to use report-specific directory
            output_filename = f"{report_name}_analysis.json"
            output_path = os.path.join(report_output_dir, output_filename)
            save_results_to_json(results, output_path)
            
            # Generate visualizations in report-specific charts directory
            create_comparison_charts(results, charts_dir)
            llm_chart_path = create_llm_comparison_chart(llm_results, charts_dir, timestamp)
            coherence_chart_path = create_coherence_chart(results, charts_dir, timestamp)
            weighted_scores_chart_path = create_weighted_scores_chart(weighted_scores, charts_dir, timestamp)
            
            print(f"\nResults for {report_name} saved to {report_output_dir}")
            print(f"Charts saved to {charts_dir}")
            
        except Exception as e:
            print(f"Error processing report {report_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate and display average scores
    if original_overall_scores and improved_overall_scores:
        avg_original = sum(original_overall_scores) / len(original_overall_scores)
        avg_improved = sum(improved_overall_scores) / len(improved_overall_scores)
        improvement = avg_improved - avg_original
        percent_improvement = (improvement / avg_original) * 100 if avg_original > 0 else 0
        
        print("\nFINAL AVERAGES ACROSS ALL REPORTS")
        print("=================================")
        print(f"Average Original Score: {avg_original}")
        print(f"Average Improved Score: {avg_improved}")
        print(f"Average Improvement: {improvement} points ({percent_improvement}%)")
        
        # Save the averages to a summary JSON file
        summary = {
            "number_of_reports": len(json_files),
            "average_scores": {
                "original": avg_original,
                "improved": avg_improved,
                "improvement": improvement,
                "percent_improvement": percent_improvement
            },
            "individual_scores": {
                "original": original_overall_scores,
                "improved": improved_overall_scores
            }
        }
        
        summary_path = os.path.join(base_output_dir, "overall_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
    
    print("\nEvaluation complete. Results organized by report in the 'evaluation_results' directory.")