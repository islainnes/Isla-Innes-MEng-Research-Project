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
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        # Analyze concept flow
        concept_flow = self.analyze_concept_flow(paragraphs)
        
        return {
            'concept_flow': concept_flow
        }
        
    def analyze_concept_flow(self, paragraphs):
        """Analyze how concepts flow with ideal ranges rather than maximums"""
        if len(paragraphs) < 2:
            return {'flow_score': 0, 'quality': 'insufficient content', 'details': {}}
        
        # Encode paragraphs
        embeddings = self.encoder.encode(paragraphs)
        
        # Calculate local coherence
        local_scores = []
        for i in range(len(paragraphs) - 1):
            similarity = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
            local_scores.append(float(similarity))
        
        # Define ideal ranges for different metrics
        ideal_ranges = {
            'local_coherence': (0.3, 0.7),  # Some similarity but not too much
            'global_coherence': (0.4, 0.8),  # Consistent theme while allowing development
            'transition_density': (0.2, 0.4)  # Enough transitions without overuse
        }
        
        # Score based on distance from ideal range
        def score_metric(value, ideal_min, ideal_max):
            if value < ideal_min:
                # Penalize being too low
                return 1 - ((ideal_min - value) / ideal_min)
            elif value > ideal_max:
                # Penalize being too high
                return 1 - ((value - ideal_max) / (1 - ideal_max))
            else:
                # Perfect score if within range
                return 1.0
        
        # Calculate component scores
        avg_local = np.mean(local_scores)
        local_quality = score_metric(avg_local, *ideal_ranges['local_coherence'])
        
        # Evaluate progression
        progression_scores = []
        for i in range(len(paragraphs) - 2):
            # Look at three consecutive paragraphs
            score1 = local_scores[i]
            score2 = local_scores[i + 1]
            # Good progression should show some variation
            variation = abs(score1 - score2)
            progression_scores.append(1.0 if 0.1 <= variation <= 0.4 else 0.5)
        
        # Qualitative assessment
        def get_quality_label(score):
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
        
        # Detailed analysis
        details = {
            'local_coherence': {
                'score': float(local_quality),
                'raw_value': float(avg_local),
                'assessment': get_quality_label(local_quality),
                'issues': []
            },
            'progression': {
                'score': float(np.mean(progression_scores)),
                'assessment': get_quality_label(np.mean(progression_scores)),
                'issues': []
            }
        }
        
        # Identify specific issues
        if avg_local < ideal_ranges['local_coherence'][0]:
            details['local_coherence']['issues'].append(
                "Paragraphs are too disconnected. Consider adding more transitions and maintaining thematic links."
            )
        elif avg_local > ideal_ranges['local_coherence'][1]:
            details['local_coherence']['issues'].append(
                "Paragraphs are too similar. Consider developing ideas more and reducing redundancy."
            )
        
        if np.mean(progression_scores) < 0.6:
            details['progression']['issues'].append(
                "Ideas aren't developing enough between paragraphs. Consider how each paragraph advances the discussion."
            )
        
        # Calculate final score with emphasis on balanced progression
        final_score = (
            0.4 * local_quality +
            0.6 * np.mean(progression_scores)  # Weight progression more heavily
        )
        
        return {
            'flow_score': float(final_score),
            'quality': get_quality_label(final_score),
            'details': details
        }

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

def count_actionable_recommendations(text):
    # Find sentences that appear to be actionable recommendations
    recommendation_patterns = [
        r'(?:should|must|need to|recommend|advise|suggest).*?[.!]',  # Directive language
        r'(?:it is recommended|we recommend|I recommend).*?[.!]',  # Explicit recommendations
        r'(?:consider|implement|adopt|develop|establish|create).*?[.!]'  # Action verbs
    ]
    
    # Combine patterns and search for matches
    combined_pattern = '|'.join(recommendation_patterns)
    recommendations = re.findall(combined_pattern, text, re.IGNORECASE)
    
    return len(recommendations)

def count_technical_terms(text):
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
    
    # Count occurrences of each term
    term_count = 0
    for term in technical_terms:
        # Use word boundaries to ensure we're matching whole words
        term_count += len(re.findall(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
    
    return term_count

def estimate_concept_hierarchy_depth(text):
    """
    Estimates the hierarchical depth of concepts in text using topic modeling
    and syntactic structure analysis.
    """
    # Initialize components
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        # If model not installed, download it
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        nlp = spacy.load('en_core_web_sm')

    # Part 1: Topic Hierarchy Analysis
    topic_depth = analyze_topic_hierarchy(text)
    
    # Part 2: Sentence Complexity Analysis
    syntax_depth = analyze_sentence_complexity(text, nlp)
    
    # Combine scores
    final_depth = calculate_hierarchy_score(topic_depth, syntax_depth)
    
    return final_depth

def analyze_topic_hierarchy(text, num_topics=5, num_words=10):
    """
    Uses LDA to identify topic hierarchy levels
    """
    # Preprocess text
    sentences = text.split('.')
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english'
    )
    doc_term_matrix = vectorizer.fit_transform(sentences)
    
    # Apply LDA
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    lda.fit(doc_term_matrix)
    
    # Analyze topic distribution
    topic_distributions = lda.transform(doc_term_matrix)
    
    # Calculate topic hierarchy depth based on:
    # 1. Number of significant topics (topics with distribution > threshold)
    # 2. Topic distinctiveness
    threshold = 0.1
    significant_topics = np.sum(np.max(topic_distributions, axis=1) > threshold)
    
    # Calculate topic distinctiveness
    topic_distinctiveness = np.mean(np.std(topic_distributions, axis=1))
    
    # Combine metrics to estimate topic hierarchy depth
    topic_depth = (significant_topics * topic_distinctiveness) / 2
    
    return min(max(1, round(topic_depth)), 5)  # Bound between 1 and 5

def analyze_sentence_complexity(text, nlp):
    """
    Uses spaCy to analyze syntactic complexity through dependency parsing
    """
    doc = nlp(text)
    
    # Calculate average dependency tree depth for each sentence
    depths = []
    for sent in doc.sents:
        # Create dependency tree
        tree_depths = []
        for token in sent:
            depth = 1
            current = token
            while current.head != current:  # Walk up the tree until we hit the root
                depth += 1
                current = current.head
            tree_depths.append(depth)
        
        if tree_depths:
            depths.append(max(tree_depths))
    
    # Calculate average depth across all sentences
    avg_depth = np.mean(depths) if depths else 1
    
    # Normalize to a 1-5 scale
    normalized_depth = min(max(1, round(avg_depth / 2)), 5)
    
    return normalized_depth

def calculate_hierarchy_score(topic_depth, syntax_depth):
    """
    Combines topic and syntactic hierarchy scores into final depth score
    """
    # Weight topic depth slightly more than syntax depth
    weighted_score = (topic_depth * 0.6) + (syntax_depth * 0.4)
    
    # Round to nearest integer and ensure bounds
    final_score = min(max(1, round(weighted_score)), 5)
    
    return final_score

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
    Use an LLM (Claude) to evaluate the original and improved reports
    Returns the evaluation results
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # Create a combined prompt to evaluate both texts
    prompt = f"""I'll provide you with two versions of a report: an original version and an improved version.
    Please evaluate both versions for technical depth, clarity, and overall effectiveness.
    
    Provide a score from 0-100 for each category, with justification and specific observations.
    
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
                "score": <0-100>,
                "justification": "detailed explanation"
            }},
            "clarity": {{
                "score": <0-100>,
                "justification": "detailed explanation"
            }},
            "overall": {{
                "score": <0-100>,
                "justification": "detailed explanation"
            }}
        }},
        "improved": {{
            "technical_depth": {{
                "score": <0-100>,
                "justification": "detailed explanation"
            }},
            "clarity": {{
                "score": <0-100>,
                "justification": "detailed explanation"
            }},
            "overall": {{
                "score": <0-100>,
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
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the JSON response from Claude's message
        response_text = response.content[0].text
        
        # Find the JSON response in the text (in case Claude adds additional commentary)
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
                'technical_depth': {'score': 50, 'justification': "Error in evaluation process"},
                'clarity': {'score': 50, 'justification': "Error in evaluation process"},
                'overall': {'score': 50, 'justification': "Error in evaluation process"}
            },
            'improved': {
                'technical_depth': {'score': 50, 'justification': "Error in evaluation process"},
                'clarity': {'score': 50, 'justification': "Error in evaluation process"},
                'overall': {'score': 50, 'justification': "Error in evaluation process"}
            },
            'comparison': {
                'technical_depth_difference': 0,
                'clarity_difference': 0,
                'overall_difference': 0,
                'summary': "Error in evaluation process"
            }
        }

def create_llm_comparison_chart(llm_results, output_dir, timestamp):
    """
    Create a bar chart comparing LLM evaluation scores
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
    plt.ylabel('Score (0-100)')
    plt.xticks(x, labels)
    plt.title('LLM Evaluation Scores Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(original_values):
        plt.text(i - width/2, v + 3, str(v), ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(improved_values):
        plt.text(i + width/2, v + 3, str(v), ha='center', va='bottom', fontweight='bold')
    
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
                f"{diff:+}",
                xy=(arrow_x, arrow_y_start),
                xytext=(arrow_x, arrow_y_start - 10 if diff < 0 else arrow_y_start + 10),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2),
                ha='center',
                va='center',
                fontweight='bold',
                color=arrow_color
            )
    
    # Set y-axis limit to 0-100 with some padding
    plt.ylim(0, 105)
    
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

def calculate_weighted_score(report_metrics, llm_results):
    """
    Calculate a weighted score (0-100) based on various metrics
    
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
        'coherence_flow_score': 0.15,  # 15%
        'actionable_recommendations_count': 0.05,  # 5%
    }

    # Calculate normalized scores (0-100 scale)
    scores = {
        # Technical Depth
        'technical_term_count': min(report_metrics['technical_term_count'] * 5, 100),
        'concept_hierarchy_depth': min(report_metrics['concept_hierarchy_depth'] * 20, 100),
        'llm_technical_depth': llm_results['technical_depth']['score'],
        
        # Clarity & Understandability
        'flesch_score': min(max(report_metrics['flesch_score'], 0), 100),
        'defined_terms_count': min(report_metrics['defined_terms_count'] * 10, 100),
        'example_count': min(report_metrics['example_count'] * 20, 100),
        'llm_clarity': llm_results['clarity']['score'],
        
        # Structure
        'coherence_flow_score': report_metrics['contextual_coherence']['concept_flow']['flow_score'] * 100,
        'actionable_recommendations_count': min(report_metrics['actionable_recommendations_count'] * 20, 100)
    }
    
    # Calculate weighted scores
    technical_score = sum(scores[metric] * weight for metric, weight in technical_weights.items())
    clarity_score = sum(scores[metric] * weight for metric, weight in clarity_weights.items())
    structure_score = sum(scores[metric] * weight for metric, weight in structure_weights.items())
    
    # Calculate final score
    final_score = technical_score + clarity_score + structure_score
    
    return {
        'final_score': round(final_score, 2),
        'component_scores': {
            'technical_depth': round(technical_score, 2),
            'clarity': round(clarity_score, 2),
            'structure': round(structure_score, 2)
        },
        'detailed_scores': {metric: round(score, 2) for metric, score in scores.items()}
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
    plt.ylabel('Score (0-100)')
    plt.title('Weighted Score Comparison')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(original_values):
        plt.text(i - width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(improved_values):
        plt.text(i + width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
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
                f"{diff:+.1f}",
                xy=(arrow_x, arrow_y_start),
                xytext=(arrow_x, arrow_y_start - 5 if diff < 0 else arrow_y_start + 5),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2),
                ha='center',
                va='center',
                fontweight='bold',
                color=arrow_color
            )
    
    # Add percentage improvement as a subtitle
    plt.figtext(
        0.5, 0.02,
        f"Overall Improvement: {weighted_scores['percent_improvement']:+.1f}%",
        ha='center',
        fontsize=10,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )
    
    # Set y-axis limit to 0-100 with some padding
    plt.ylim(0, 105)
    
    # Save the chart
    chart_path = os.path.join(output_dir, f"weighted_scores_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Created weighted scores chart: {chart_path}")
    plt.close()
    
    return chart_path

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
            original_recommendations = count_actionable_recommendations(original_report)
            original_technical_terms = count_technical_terms(original_report)
            original_examples = count_examples(original_report)
            original_defined_terms = count_defined_terms(original_report)
            
            # Calculate technical metrics for improved report
            improved_concept_depth = estimate_concept_hierarchy_depth(improved_report)
            improved_recommendations = count_actionable_recommendations(improved_report)
            improved_technical_terms = count_technical_terms(improved_report)
            improved_examples = count_examples(improved_report)
            improved_defined_terms = count_defined_terms(improved_report)
            
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get LLM evaluation
            print("\nRequesting LLM evaluation...")
            llm_results = llm_evaluate_report(original_report, improved_report, os.environ.get('ANTHROPIC_API_KEY') or "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")
            
            # Create a results dictionary including LLM evaluation
            results = {
                "file_analyzed": json_path,
                "timestamp": timestamp,
                "original_report": {
                    "word_count": original_length,
                    "flesch_score": round(original_score, 2),
                    "concept_hierarchy_depth": original_concept_depth,
                    "actionable_recommendations_count": original_recommendations,
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
                    "concept_hierarchy_depth": improved_concept_depth,
                    "actionable_recommendations_count": improved_recommendations,
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
                    "concept_depth_difference": improved_concept_depth - original_concept_depth,
                    "recommendations_difference": improved_recommendations - original_recommendations,
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
            print(f"Concept hierarchy depth: {original_concept_depth} → {improved_concept_depth}")
            print(f"Actionable recommendations: {original_recommendations} → {improved_recommendations}")
            print(f"Technical terms: {original_technical_terms} → {improved_technical_terms}")
            
            # Print LLM evaluation
            print("\nLLM EVALUATION")
            print("==============")
            print("Original Report:")
            print(f"Technical Depth: {llm_results['original']['technical_depth']['score']}/100")
            print(f"Clarity: {llm_results['original']['clarity']['score']}/100")
            print(f"Overall: {llm_results['original']['overall']['score']}/100")
            
            print("\nImproved Report:")
            print(f"Technical Depth: {llm_results['improved']['technical_depth']['score']}/100")
            print(f"Clarity: {llm_results['improved']['clarity']['score']}/100")
            print(f"Overall: {llm_results['improved']['overall']['score']}/100")
            
            print("\nComparison:")
            print(f"Technical Depth Difference: {llm_results['comparison']['technical_depth_difference']:+}")
            print(f"Clarity Difference: {llm_results['comparison']['clarity_difference']:+}")
            print(f"Overall Difference: {llm_results['comparison']['overall_difference']:+}")
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
            print(f"Original Report: {weighted_scores['original']['final_score']}/100")
            print(f"Improved Report: {weighted_scores['improved']['final_score']}/100")
            print(f"Improvement: {weighted_scores['difference']:+} points ({weighted_scores['percent_improvement']:+.2f}%)")

            print("\nComponent Improvements:")
            print(f"Technical Depth: {weighted_scores['component_differences']['technical_depth']:+.2f}")
            print(f"Clarity: {weighted_scores['component_differences']['clarity']:+.2f}")
            print(f"Structure: {weighted_scores['component_differences']['structure']:+.2f}")
            
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
        print(f"Average Original Score: {avg_original:.2f}/100")
        print(f"Average Improved Score: {avg_improved:.2f}/100")
        print(f"Average Improvement: {improvement:+.2f} points ({percent_improvement:+.2f}%)")
        
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
