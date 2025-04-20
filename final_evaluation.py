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
    
    # Calculate Gunning Fog Index scores
    original_score = textstat.gunning_fog(original_report)
    improved_score = textstat.gunning_fog(improved_report)
    
    # Get technical metrics for original and improved reports
    original_tech_metrics = count_technical_terms_with_ner(original_report)
    improved_tech_metrics = count_technical_terms_with_ner(improved_report)
    
    # Calculate technical depth, clarity and structure for both reports
    original_technical_depth = calculate_technical_depth(original_report)
    improved_technical_depth = calculate_technical_depth(improved_report)
    
    original_clarity = calculate_clarity(original_report)
    improved_clarity = calculate_clarity(improved_report)
    
    original_structure = calculate_structure(original_report)
    improved_structure = calculate_structure(improved_report)
    
    # Extract balanced technical scores
    original_balanced_score = original_tech_metrics.get('balanced_technical_score', 0)
    improved_balanced_score = improved_tech_metrics.get('balanced_technical_score', 0)
    
    return (
        original_report, 
        improved_report, 
        original_score, 
        improved_score, 
        original_clarity.get('coherence', {}),  # Coherence now in clarity metrics
        improved_clarity.get('coherence', {}),  # Coherence now in clarity metrics 
        original_balanced_score, 
        improved_balanced_score,
        original_technical_depth,
        improved_technical_depth,
        original_clarity,
        improved_clarity,
        original_structure,
        improved_structure
    )

def count_technical_terms_with_ner(text):
    """
    Count and identify technical terms using a comprehensive semiconductor dictionary
    first, then supplement with NLP techniques. Returns both raw counts and
    frequency normalized to 0-1 scale using predefined ranges based on document length.
    
    The implementation explicitly uses both dictionary-based and NER-based approaches
    with balanced scoring to provide a comprehensive technical terminology analysis.
    """
    # Comprehensive dictionary of semiconductor and electronics terminology
    semiconductor_dictionary = {
        # Materials
        "silicon", "germanium", "gallium arsenide", "gaas", "gan", "sic", "silicon carbide",
        "semiconductor", "dielectric", "oxide", "nitride", "polysilicon", "silicon dioxide", "sio2",
        "silicon nitride", "si3n4", "high-k", "low-k", "copper", "aluminum", "tungsten", "titanium",
        "tantalum", "cobalt", "silicide", "germanium", "iii-v", "ii-vi", "compound semiconductor",
        "heterojunction", "quantum well", "quantum dot", "superlattice", "nanowire", "graphene", 
        "2d materials", "perovskite", "organic semiconductor",
        
        # Devices
        "transistor", "mosfet", "fet", "bjt", "igbt", "thyristor", "diode", "led", "photodiode",
        "phototransistor", "cmos", "nmos", "pmos", "hemt", "mesfet", "jfet", "pin diode", "schottky",
        "varactor", "solar cell", "pv cell", "memory cell", "capacitor", "resistor", "inductor",
        "memristor", "sensor", "mems", "nems", "integrated circuit", "ic", "chip", "die",
        "power device", "logic gate", "amplifier", "oscillator", "flip-flop", "latch", "register",
        
        # Properties & Physics
        "bandgap", "band gap", "energy band", "conduction band", "valence band", "fermi level",
        "doping", "dopant", "n-type", "p-type", "carrier", "electron", "hole", "mobility",
        "conductivity", "resistivity", "junction", "depletion region", "inversion layer",
        "threshold voltage", "breakdown voltage", "leakage current", "saturation current",
        "channel", "source", "drain", "gate", "substrate", "body effect", "pinch-off",
        "avalanche breakdown", "tunneling", "quantum tunneling", "ballistic transport",
        "scattering", "phonon", "recombination", "generation", "lifetime", "diffusion",
        "drift", "carrier lifetime", "minority carrier", "majority carrier", "band bending",
        "work function", "electron affinity", "interface state", "trap", "defect",
        
        # Fabrication & Processing
        "fabrication", "wafer", "epitaxy", "cvd", "pecvd", "mocvd", "mbe", "ald", "pvd",
        "sputtering", "evaporation", "lithography", "photolithography", "euv", "e-beam",
        "photoresist", "mask", "stepper", "scanner", "etch", "etching", "wet etch", "dry etch",
        "rie", "plasma", "implantation", "ion implantation", "diffusion", "oxidation",
        "annealing", "rta", "chemical mechanical polishing", "cmp", "metallization",
        "interconnect", "backend", "frontend", "cleaning", "deposition", "thin film",
        "photomask", "reticle", "alignment", "overlay", "damascene", "dual damascene",
        "planarization", "sintering", "bonding", "packaging", "dicing", "wire bonding",
        "flip chip", "passivation", "gettering", "thermal budget",
        
        # Characterization & Testing
        "characterization", "metrology", "sem", "tem", "afm", "stm", "xrd", "sims", "xps",
        "ellipsometry", "profilometer", "four-point probe", "hall measurement", "c-v",
        "capacitance-voltage", "i-v", "current-voltage", "reliability", "lifetime",
        "failure mechanism", "electromigration", "stress migration", "tddb", "hot carrier",
        "nbti", "pbti", "testing", "probe", "probe card", "wafer testing", "parametric test",
        "functional test", "iddq", "burn-in", "yield", "defect density", "statistical analysis",
        
        # Circuits & Design
        "circuit", "analog", "digital", "mixed-signal", "rf", "microwave", "integrated circuit",
        "vlsi", "uv", "euv", "layout", "schematic", "netlist", "spice", "simulation", "parasitic",
        "eda", "cad", "verification", "timing", "power", "leakage", "dynamic power", "static power",
        "signal integrity", "noise margin", "fan-out", "fan-in", "standard cell", "ip block",
        "soc", "system-on-chip", "asic", "fpga", "pld", "memory", "ram", "dram", "sram",
        "flash", "rom", "embedded", "peripherals", "processor", "cpu", "gpu", "dsp",
        "adc", "dac", "pll", "dlx", "lna", "mixer", "filter",
        
        # Advanced Topics
        "quantum computing", "spintronics", "photonics", "silicon photonics", "neuromorphic",
        "memristor", "reram", "mram", "pram", "feram", "emerging memory", "3d integration",
        "heterogeneous integration", "chiplet", "tsv", "through-silicon via", "2.5d", "3d ic",
        "packaging", "fan-out", "embedded die", "wafer-level", "rf-soi", "fd-soi", "finfet",
        "gaafet", "nanosheet", "nanowire", "negative capacitance", "tunnel fet", "tfet",
        "wide bandgap", "ultra-wide bandgap", "gan", "sic", "diamond", "uv led"
    }
    
    # Create case-insensitive version (convert all to lowercase)
    semiconductor_dictionary = {term.lower() for term in semiconductor_dictionary}
    
    # Initialize containers for tracking terms by source
    dictionary_terms = []  # Terms found via dictionary
    ner_terms = []         # Terms found via NER
    combined_terms = []    # All terms (combined approach)
    
    total_words = len(text.split())
    
    # 1. Dictionary-based identification
    # Tokenize text for basic word-level matching
    words = re.findall(r'\b[a-zA-Z0-9][\w\-\.]*[a-zA-Z0-9]\b|\b[a-zA-Z0-9]\b', text.lower())
    
    # Check single words against dictionary
    for word in words:
        if word.lower() in semiconductor_dictionary:
            dictionary_terms.append(word.lower())
            combined_terms.append(word.lower())
    
    # Check for multi-word terms from the dictionary
    for term in semiconductor_dictionary:
        if ' ' in term and term.lower() in text.lower():
            # Count each occurrence
            count = text.lower().count(term.lower())
            for _ in range(count):
                dictionary_terms.append(term.lower())
                combined_terms.append(term.lower())
    
    # 2. Supplement with NLP techniques
    try:
        # Load spaCy model - scientific models work best but fall back to standard if needed
        try:
            nlp = spacy.load('en_core_sci_md')  # Scientific model for better technical term detection
        except OSError:
            try:
                nlp = spacy.load('en_core_web_lg')  # Fall back to standard large model
            except OSError:
                import subprocess
                print("Downloading spaCy model...")
                subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_lg'])
                nlp = spacy.load('en_core_web_lg')
        
        # Process the text
        doc = nlp(text)
        
        # 2.1. Get named entities that are likely technical
        for ent in doc.ents:
            # Focus on organization, product, and other relevant entity types 
            # which often capture technical concepts
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LAW', 'WORK_OF_ART']:
                term = ent.text.lower()
                if term not in semiconductor_dictionary:  # Only add if not already in dictionary
                    ner_terms.append(term)
                    combined_terms.append(term)
        
        # 2.2. Add technical noun chunks (multi-word technical terms)
        technical_adjectives = [
            'quantum', 'electrical', 'electronic', 'thermal', 'optical', 'solar', 
            'semiconductor', 'transistor', 'bipolar', 'diode', 'integrated', 'digital',
            'analog', 'rf', 'microwave', 'photonic', 'ionic', 'ferroelectric', 'ferromagnetic',
            'superconducting', 'piezoelectric', 'electrostatic', 'electromagnetic',
            'photovoltaic', 'optoelectronic', 'nanoscale', 'microscale', 'high-frequency',
            'low-power', 'high-voltage', 'single-crystal', 'polycrystalline', 'amorphous'
        ]
        
        for chunk in doc.noun_chunks:
            # If noun chunk contains technical adjectives
            if any(token.text.lower() in technical_adjectives for token in chunk):
                term = chunk.text.lower()
                if term not in semiconductor_dictionary:  # Only add if not already in dictionary
                    ner_terms.append(term)
                    combined_terms.append(term)
        
        # 2.3. Pattern matching for chemical formulas, measurements, etc.
        chemical_formula_pattern = r'\b[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)+\b'
        measurement_pattern = r'\b\d+(?:\.\d+)?(?:n|µ|m|k|M|G)?(?:m|A|V|W|Hz|eV|Ω|F|H)\b'
        
        chemical_formulas = re.findall(chemical_formula_pattern, text)
        for formula in chemical_formulas:
            if formula.lower() not in semiconductor_dictionary:
                ner_terms.append(formula.lower())
                combined_terms.append(formula.lower())
                
        measurements = re.findall(measurement_pattern, text)
        for measurement in measurements:
            # Don't add simple numbers as technical terms
            if any(unit in measurement for unit in ['m', 'A', 'V', 'W', 'Hz', 'eV', 'Ω', 'F', 'H']):
                ner_terms.append(measurement.lower())
                combined_terms.append(measurement.lower())
        
    except Exception as e:
        print(f"Error in NLP-based technical term analysis: {str(e)}")
    
    # Calculate metrics for each approach
    dictionary_count = len(dictionary_terms)
    ner_count = len(ner_terms)
    combined_count = len(combined_terms)
    
    # Create term frequency dictionaries
    dictionary_frequencies = {}
    for term in dictionary_terms:
        if term in dictionary_frequencies:
            dictionary_frequencies[term] += 1
        else:
            dictionary_frequencies[term] = 1
    
    ner_frequencies = {}
    for term in ner_terms:
        if term in ner_frequencies:
            ner_frequencies[term] += 1
        else:
            ner_frequencies[term] = 1
    
    combined_frequencies = {}
    for term in combined_terms:
        if term in combined_frequencies:
            combined_frequencies[term] += 1
        else:
            combined_frequencies[term] = 1
    
    # Get unique terms for each approach
    unique_dictionary_terms = set(dictionary_terms)
    unique_ner_terms = set(ner_terms)
    unique_combined_terms = set(combined_terms)
    
    # Calculate technical frequency using combined raw count
    technical_frequency = combined_count / max(1, total_words)
    technical_percentage = technical_frequency * 100  # Convert to percentage for readability
    
    # Calculate percentage of dictionary covered
    dictionary_terms_found = set()
    for term in unique_combined_terms:
        if term in semiconductor_dictionary:
            dictionary_terms_found.add(term)
    
    dictionary_coverage_percentage = (len(dictionary_terms_found) / len(semiconductor_dictionary)) * 100
    
    # Determine document length category for context
    if total_words < 200:
        doc_length_category = "very short"
    elif total_words < 500:
        doc_length_category = "short-medium"
    elif total_words < 1000:
        doc_length_category = "medium"
    elif total_words < 3000:
        doc_length_category = "medium-long"
    else:
        doc_length_category = "long"
    
    # Normalize using predefined ranges based on document length category
    # Similar to how Flesch score is normalized with predefined categories
    if doc_length_category == "very short":  # < 200 words
        if technical_percentage < 5:
            normalized_score = 0.2  # Minimal technical content
        elif technical_percentage < 10:
            normalized_score = 0.4  # Low technical content
        elif technical_percentage < 15:
            normalized_score = 0.6  # Moderate technical content
        elif technical_percentage < 20:
            normalized_score = 0.8  # High technical content
        else:
            normalized_score = 1.0  # Very high technical content
    
    elif doc_length_category == "short-medium":  # 200-499 words
        if technical_percentage < 4:
            normalized_score = 0.2  # Minimal technical content
        elif technical_percentage < 8:
            normalized_score = 0.4  # Low technical content
        elif technical_percentage < 12:
            normalized_score = 0.6  # Moderate technical content
        elif technical_percentage < 18:
            normalized_score = 0.8  # High technical content
        else:
            normalized_score = 1.0  # Very high technical content
    
    elif doc_length_category == "medium":  # 500-999 words
        if technical_percentage < 3:
            normalized_score = 0.2  # Minimal technical content
        elif technical_percentage < 6:
            normalized_score = 0.4  # Low technical content
        elif technical_percentage < 10:
            normalized_score = 0.6  # Moderate technical content
        elif technical_percentage < 15:
            normalized_score = 0.8  # High technical content
        else:
            normalized_score = 1.0  # Very high technical content
            
    elif doc_length_category == "medium-long":  # 1000-2999 words
        if technical_percentage < 2:
            normalized_score = 0.2  # Minimal technical content
        elif technical_percentage < 5:
            normalized_score = 0.4  # Low technical content
        elif technical_percentage < 8:
            normalized_score = 0.6  # Moderate technical content
        elif technical_percentage < 13:
            normalized_score = 0.8  # High technical content
        else:
            normalized_score = 1.0  # Very high technical content
            
    else:  # long (3000+ words)
        if technical_percentage < 1.5:
            normalized_score = 0.2  # Minimal technical content
        elif technical_percentage < 4:
            normalized_score = 0.4  # Low technical content
        elif technical_percentage < 7:
            normalized_score = 0.6  # Moderate technical content
        elif technical_percentage < 10:
            normalized_score = 0.8  # High technical content
        else:
            normalized_score = 1.0  # Very high technical content
    
    # Calculate a balanced score between dictionary and NER approaches
    # Weight dictionary terms slightly higher (60%) as they are more reliable indicators
    dictionary_weight = 0.60
    ner_weight = 0.40
    
    # Normalize each count by total words for fair comparison
    normalized_dictionary_count = dictionary_count / max(1, total_words)
    normalized_ner_count = ner_count / max(1, total_words)
    
    # Calculate weighted score
    balanced_technical_score = (
        (dictionary_weight * normalized_dictionary_count) + 
        (ner_weight * normalized_ner_count)
    ) * 100  # Convert to percentage
    
    # Log the normalization details for debugging
    normalization_details = {
        'document_length': total_words,
        'document_category': doc_length_category,
        'technical_term_count': combined_count,
        'technical_frequency': technical_frequency,
        'technical_percentage': technical_percentage,
        'normalized_score': normalized_score,
        'dictionary_coverage_percentage': dictionary_coverage_percentage,
        'balanced_technical_score': balanced_technical_score,
        'dictionary_weight': dictionary_weight,
        'ner_weight': ner_weight
    }
    
    return {
        'raw_count': combined_count,
        'unique_terms': len(unique_combined_terms),
        'term_frequencies': combined_frequencies,
        'technical_frequency': technical_frequency,
        'normalized_score': normalized_score,
        'total_words': total_words,
        'identified_terms': list(unique_combined_terms),  # Include the actual terms for transparency
        'dictionary_coverage_percentage': dictionary_coverage_percentage,
        'dictionary_terms_found': list(dictionary_terms_found),
        'total_dictionary_terms': len(semiconductor_dictionary),
        'dictionary_metrics': {
            'raw_count': dictionary_count,
            'unique_terms': len(unique_dictionary_terms),
            'term_frequencies': dictionary_frequencies,
        },
        'ner_metrics': {
            'raw_count': ner_count,
            'unique_terms': len(unique_ner_terms),
            'term_frequencies': ner_frequencies,
        },
        'balanced_technical_score': balanced_technical_score,
        'normalization_details': normalization_details
    }

def estimate_concept_hierarchy_depth(text):
    """
    Estimates the hierarchical depth of concepts in text using topic modeling
    and syntactic structure analysis, returning scores on a 0-1 scale.
    
    The analysis includes:
    1. Topic hierarchy - measures conceptual organization using topic modeling
    2. Syntax complexity - measures linguistic structure via dependency parsing,
       focusing on tree depth and complex clause usage
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
        # 1. Topic diversity: measure how evenly topics are distributed across the entire document
        # Aggregate topic distribution across all sentences
        global_topic_dist = np.mean(topic_distributions, axis=0)
        
        # Calculate entropy (higher = more even distribution)
        from scipy.stats import entropy
        topic_evenness = entropy(global_topic_dist)
        
        # Normalize to 0-1 scale
        max_entropy = np.log(num_topics)  # Theoretical maximum entropy
        normalized_diversity = min(topic_evenness / max_entropy, 1.0)
        
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
        
        # 2. Complex clause usage - count subordinate clauses
        clause_count = len([token for token in doc if token.dep_ in ('ccomp', 'xcomp', 'advcl')])
        normalized_clauses = min(clause_count / (len(list(doc.sents)) * 1.5), 1.0)  # Normalize per sentence
        
        # Combined score with revised weights:
        # - Tree depth: 70% (increased from 50%)
        # - Complex clause usage: 30% (increased from 20%)
        # - Variety: removed (was 30%)
        combined_score = (
            0.7 * normalized_depth +
            0.3 * normalized_clauses
        )
        
        return combined_score
        
    except Exception as e:
        print(f"Error in syntax analysis: {e}")
        return 0.4  # Default fallback score

def count_examples(text):
    # This function is now deprecated in favor of LLM-based evaluation
    print("Warning: count_examples is deprecated. Use evaluate_definitions_and_examples_with_llm instead.")
    return 0

def count_defined_terms(text):
    # This function is now deprecated in favor of LLM-based evaluation
    print("Warning: count_defined_terms is deprecated. Use evaluate_definitions_and_examples_with_llm instead.")
    return 0

def detect_definitions_enhanced(doc, technical_terms=None):
    # This function is now deprecated in favor of LLM-based evaluation
    print("Warning: detect_definitions_enhanced is deprecated. Use evaluate_definitions_and_examples_with_llm instead.")
    return [], set()

def detect_examples_enhanced(doc):
    # This function is now deprecated in favor of LLM-based evaluation
    print("Warning: detect_examples_enhanced is deprecated. Use evaluate_definitions_and_examples_with_llm instead.")
    return []

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
    metrics = ['word_count', 'gunning_fog_score']
    labels = ['Word Count', 'Gunning Fog Index']
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
        'dictionary_coverage_percentage',  
        'balanced_technical_score',  # Added balanced technical score
        'example_count', 
        'defined_terms_count'
    ]
    
    tech_labels = [
        'Concept Hierarchy\nDepth',
        'Actionable\nRecommendations', 
        'Technical\nTerms',
        'Dictionary\nCoverage (%)',  
        'Balanced Technical\nScore (%)',  # Added balanced technical score label
        'Examples', 
        'Defined\nTerms'
    ]
    
    x = np.arange(len(tech_metrics))
    
    # Handle potential missing metrics in existing results
    if 'dictionary_coverage_percentage' not in original:
        original['dictionary_coverage_percentage'] = 0
    if 'dictionary_coverage_percentage' not in improved:
        improved['dictionary_coverage_percentage'] = 0
    if 'balanced_technical_score' not in original:
        original['balanced_technical_score'] = 0
    if 'balanced_technical_score' not in improved:
        improved['balanced_technical_score'] = 0
    
    original_tech_values = [original.get(m, 0) for m in tech_metrics]
    improved_tech_values = [improved.get(m, 0) for m in tech_metrics]
    
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
                 str(round(v, 2)) if isinstance(v, float) else str(v), ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(improved_tech_values):
        plt.text(i + width/2, v + max(original_tech_values + improved_tech_values)*0.02,
                 str(round(v, 2)) if isinstance(v, float) else str(v), ha='center', va='bottom', fontweight='bold')
    
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
    gunning_fog_change = (results["comparison"]["gunning_fog_difference"] / original["gunning_fog_score"]) * 100 if original["gunning_fog_score"] != 0 else 0
    
    # Technical metrics
    concept_depth_change = results["comparison"]["concept_depth_difference"]
    recommendations_change = results["comparison"]["recommendations_difference"]
    tech_terms_change = results["comparison"]["technical_terms_difference"]
    
    # Dictionary coverage change
    if "dictionary_coverage_percentage" in original and "dictionary_coverage_percentage" in improved:
        dictionary_coverage_change = improved["dictionary_coverage_percentage"] - original["dictionary_coverage_percentage"]
    else:
        dictionary_coverage_change = 0
    
    # Balanced technical score change
    if "balanced_technical_score" in original and "balanced_technical_score" in improved:
        balanced_technical_change = improved["balanced_technical_score"] - original["balanced_technical_score"]
    else:
        balanced_technical_change = 0
    
    examples_change = results["comparison"]["examples_difference"]
    defined_terms_change = results["comparison"]["defined_terms_difference"]
    
    # Add all changes and labels
    percent_changes = [
        word_count_change, 
        gunning_fog_change, 
        concept_depth_change,
        recommendations_change, 
        tech_terms_change,
        dictionary_coverage_change,
        balanced_technical_change,  # Added balanced technical score change
        examples_change, 
        defined_terms_change
    ]
    
    metric_names = [
        'Word Count', 
        'Gunning Fog Index', 
        'Concept Depth',
        'Recommendations', 
        'Technical Terms',
        'Dictionary Coverage',
        'Balanced Technical Score',  # Added balanced technical score label
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
        results['original_report']['clarity']['coherence']['concept_flow']['flow_score']
    ]
    improved_values = [
        results['improved_report']['clarity']['coherence']['concept_flow']['flow_score']
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
    Calculate technical depth metrics using Coverage Density Index
    """
    # Get technical metrics with NER-based frequency analysis
    tech_metrics = count_technical_terms_with_ner(text)
    
    # Initialize result dictionary
    result = {
        'technical_term_metrics': tech_metrics
    }
    
    # Add LLM evaluation with better error handling
    llm_evaluation_score = 0.5  # Default score
    llm_justification = "LLM evaluation unavailable"
    
    try:
        # Fixed prompt that actually includes the text to evaluate
        prompt = f"""Rigorously evaluate the technical depth of the following text with a highly critical eye. Consider:
        1. Technical vocabulary and terminology usage
        2. Concept depth and complexity 
        3. Explanation of technical concepts
        4. Appropriate level of detail for the audience
        
        Text to evaluate:
        ```
        {text[:6000]}  # Limiting to first 6000 chars to keep within context window
        ```
        
        Provide a score from 0.0-1.0 and a brief justification.
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
        
        # Extract the JSON response with better error handling
        import re
        import json
        json_match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content)
        if json_match:
            llm_evaluation = json.loads(json_match.group(0))
            llm_evaluation_score = float(llm_evaluation.get('score', 0.5))
            llm_justification = llm_evaluation.get('justification', "No justification provided")
    except Exception as e:
        print(f"Warning: LLM evaluation failed: {str(e)}")
    
    # Always ensure llm_evaluation exists with proper structure
    result['llm_evaluation'] = {
        'score': llm_evaluation_score,
        'justification': llm_justification
    }
    
    # Calculate Coverage Density Index (CDI)
    total_words = tech_metrics['total_words']
    unique_terms = tech_metrics['unique_terms']
    
    # Calculate CDI = num_unique_technical_terms / sqrt(total_words)
    coverage_density_index = unique_terms / (total_words ** 0.5) if total_words > 0 else 0
    
    # Normalize CDI to 0-1 range for scoring
    # Assuming a reasonable range based on typical technical documents
    if coverage_density_index >= 1.5:  # Very high technical density
        cdi_score = 1.0
    elif coverage_density_index >= 1.0:
        cdi_score = 0.8
    elif coverage_density_index >= 0.7:
        cdi_score = 0.6
    elif coverage_density_index >= 0.4:
        cdi_score = 0.4
    elif coverage_density_index >= 0.2:
        cdi_score = 0.2
    else:
        cdi_score = 0.1
    
    # Store CDI information for transparency
    result['coverage_density_index'] = {
        'raw_cdi': coverage_density_index,
        'normalized_score': cdi_score,
        'unique_terms': unique_terms,
        'total_words': total_words
    }
    
    # Calculate combined score - now with just CDI and LLM evaluation
    result['combined_score'] = (
        0.5 * cdi_score +                     # CDI - increased from 15% to 50%
        0.5 * result['llm_evaluation']['score'] # LLM evaluation - increased from 30% to 50%
    )
    
    # Include details about the weighting in the result for transparency
    result['weighting_details'] = {
        'coverage_density_index_weight': 0.5,  # Increased from 0.15
        'llm_evaluation_weight': 0.5,          # Increased from 0.30
        'explanation': "Coverage Density Index assesses technical term density relative to document length, and LLM evaluation provides human-like judgment of technical depth."
    }
    
    return result

def calculate_clarity(text):
    """
    Calculate clarity and understandability metrics for a given text.
    
    Returns:
        dict: Dictionary containing clarity metrics (all normalized to 0-1):
            - gunning_fog_score: Normalized Gunning Fog Index score
            - defined_terms_count: Normalized number of defined terms
            - example_count: Normalized number of examples
            - definition_ratio: Ratio of defined technical terms (scaled)
            - example_ratio: Ratio of technical terms with examples (scaled)
            - raw_definition_ratio: Unscaled ratio of defined technical terms
            - raw_example_ratio: Unscaled ratio of technical terms with examples
            - defined_terms: List of technical terms that are defined
            - terms_with_examples: List of technical terms that have examples
            - coherence: Contextual coherence metrics (normalized to 0-1)
            - llm_evaluation: LLM-based clarity evaluation
            
    Raises:
        Exception: If any part of the analysis fails
    """
    # Calculate basic metrics
    gunning_fog = textstat.gunning_fog(text)
    
    # Get technical term metrics to use as a base for determining how many terms should be defined
    tech_metrics = count_technical_terms_with_ner(text)
    unique_technical_terms = tech_metrics['unique_terms']
    technical_terms = tech_metrics['identified_terms']

    # Normalize Gunning Fog score
    normalized_fog = normalize_gunning_fog(gunning_fog)

    # Use NER-based approach to evaluate defined terms and examples
    try:
        defined_terms_ratio, example_coverage_ratio, defined_terms, terms_with_examples = (
            evaluate_definitions_and_examples_with_ner(text, technical_terms)
        )
        
        # Get the raw ratios by recalculating
        if technical_terms:
            raw_defined_terms_ratio = len(defined_terms) / len(technical_terms)
            raw_example_coverage_ratio = len(terms_with_examples) / len(technical_terms)
        else:
            raw_defined_terms_ratio = 0.0
            raw_example_coverage_ratio = 0.0
    except Exception as e:
        print(f"Warning: NER-based analysis failed, falling back to LLM: {e}")
        # Fall back to LLM-based evaluation if NER fails
        defined_terms_ratio, example_coverage_ratio = evaluate_definitions_and_examples_with_llm(text, technical_terms)
        raw_defined_terms_ratio = defined_terms_ratio  # Without NER we don't have raw values
        raw_example_coverage_ratio = example_coverage_ratio
        defined_terms = []
        terms_with_examples = []
    
    # Calculate normalized metrics based on NER evaluation
    # For defined terms
    if defined_terms_ratio >= 0.7:  # Excellent coverage
        normalized_defined = 1.0
    elif defined_terms_ratio >= 0.5:  # Good coverage
        normalized_defined = 0.8
    elif defined_terms_ratio >= 0.3:  # Adequate coverage
        normalized_defined = 0.6
    elif defined_terms_ratio > 0:  # Some coverage
        normalized_defined = 0.4
    else:  # No coverage
        normalized_defined = 0.1
    
    # For examples
    if example_coverage_ratio >= 0.5:  # Excellent examples coverage
        normalized_examples = 1.0
    elif example_coverage_ratio >= 0.3:  # Good examples coverage
        normalized_examples = 0.8
    elif example_coverage_ratio >= 0.15:  # Adequate examples coverage
        normalized_examples = 0.6
    elif example_coverage_ratio > 0:  # Some examples
        normalized_examples = 0.4
    else:  # No examples
        normalized_examples = 0.1
    
    # Assign the variables that were missing
    defined_terms_count = normalized_defined
    example_count = normalized_examples
    definition_ratio = defined_terms_ratio
    example_ratio = example_coverage_ratio
    
    # Get coherence metrics (moved from structure calculation)
    coherence_analyzer = ContextualCoherenceAnalyzer()
    coherence = coherence_analyzer.analyze_contextual_coherence(text)
    
    # Ensure we have a valid flow score
    flow_score = coherence.get('concept_flow', {}).get('flow_score', 0.5)
    if np.isnan(flow_score):
        flow_score = 0.5  # Default to neutral score if NaN
    
    # Add LLM evaluation for overall clarity
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
            raise ValueError("Failed to parse LLM response")
            
    except Exception as e:
        raise Exception(f"LLM evaluation failed: {str(e)}")
    
    # Modify these weightings to include coherence
    gunning_fog_weight = 0.20  # Reduced from 25% to 20%
    defined_terms_weight = 0.15  # Unchanged at 15%
    example_weight = 0.15        # Unchanged at 15%
    coherence_weight = 0.20      # Added coherence at 20%
    llm_weight = 0.30           # Reduced from 45% to 30%
    
    # Ensure weights sum to 1
    assert abs(gunning_fog_weight + defined_terms_weight + example_weight + coherence_weight + llm_weight - 1.0) < 1e-10
    
    # Calculate combined score with new weightings
    combined_score = (
        gunning_fog_weight * gunning_fog +
        defined_terms_weight * defined_terms_count +
        example_weight * example_count +
        coherence_weight * flow_score +
        llm_weight * llm_evaluation["score"]
    )
    
    # Return dictionary with all metrics including raw values and identified terms
    return {
        "flesch_score": gunning_fog,
        "defined_terms_count": defined_terms_count,
        "example_count": example_count,
        "definition_ratio": definition_ratio,
        "example_ratio": example_ratio,
        "raw_definition_ratio": raw_defined_terms_ratio,
        "raw_example_ratio": raw_example_coverage_ratio,
        "coherence": coherence,
        "defined_terms": defined_terms,
        "terms_with_examples": terms_with_examples,
        "llm_evaluation": llm_evaluation,
        "combined_score": combined_score,
        "weights": {
            "gunning_fog_weight": gunning_fog_weight,
            "defined_terms_weight": defined_terms_weight,
            "example_weight": example_weight,
            "coherence_weight": coherence_weight,
            "llm_weight": llm_weight
        }
    }

def evaluate_definitions_and_examples_with_ner(text, technical_terms):
    """
    Use NER and pattern matching to determine which technical terms are defined and which have examples
    in the text. This is a more precise approach than using LLM evaluation.
    
    Args:
        text (str): The text to evaluate
        technical_terms (list): List of identified technical terms
        
    Returns:
        tuple: (defined_terms_ratio, example_coverage_ratio, defined_terms, terms_with_examples)
    """
    # Print all technical terms for debugging
    print("\n===== TECHNICAL TERMS DETECTED =====")
    for i, term in enumerate(technical_terms):
        print(f"{i+1}. {term}")
    print(f"Total technical terms: {len(technical_terms)}")
    
    # Load spaCy model
    try:
        # Try to load a scientific model first
        try:
            nlp = spacy.load('en_core_sci_md')  # Scientific model for better technical term detection
        except OSError:
            # Fall back to standard model
            nlp = spacy.load('en_core_web_lg')  # Fall back to standard large model
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        try:
            # Last resort - download and load the smaller model
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            print(f"Failed to load any spaCy model: {e}")
            # Return default values if we can't load any model
            return 0.0, 0.0, [], []
    
    # Process the text
    doc = nlp(text)
    
    # Dictionary to store terms and their status
    defined_terms = set()
    terms_with_examples = set()
    
    # Dictionary to track which patterns matched each term
    definition_matches = {}
    
    # Pattern matching for explicit definitions
    definition_patterns = [
        r'(\w+(?:\s+\w+){0,5})\s+(?:is|are|refers to|means|is defined as|can be defined as)\s+',
        r'(\w+(?:\s+\w+){0,5})\s+(?:denotes|signifies|represents|describes)\s+',
        r'(?:term|concept|notion|phrase)\s+["\']([^"\']+)["\']',
        r'(?:defining|definition of|define)\s+["\']?([^,\.]+)["\']?'
    ]
    
    # Pattern matching for examples
    example_patterns = [
        r'(?:for example|for instance|e\.g\.|such as|like)[,:]?\s*([^\.;]+)[\.;]',
        r'(?:as an example|as examples|examples include)[,:]?\s*([^\.;]+)[\.;]',
        r'(?:illustrated by|demonstrated by|exemplified by)[,:]?\s*([^\.;]+)[\.;]',
        r'(?:example[s]? of\s+)([^\.;]+)[\.;]',
        r'(\w+(?:\s+\w+){0,5})\s+(?:is an example|are examples)',
        r'["\']([^"\']+)["\'],?\s+(?:is an example|as an example)'
    ]
    
    # Print header for definition pattern matching
    print("\n===== DEFINITION PATTERN MATCHES =====")
    
    # Find defined terms using pattern matching
    for i, pattern in enumerate(definition_patterns):
        print(f"\nPattern {i+1}: {pattern}")
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            defined_term = match.group(1).lower().strip()
            print(f"  Match: '{defined_term}'")
            # Check if this matches or contains any of our technical terms
            for term in technical_terms:
                if term.lower() in defined_term or defined_term in term.lower():
                    defined_terms.add(term)
                    if term not in definition_matches:
                        definition_matches[term] = []
                    definition_matches[term].append(f"Pattern {i+1}")
                    print(f"    --> Matched technical term: '{term}'")
    
    # Find examples using pattern matching
    for pattern in example_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            example_text = match.group(1).lower().strip()
            # Check which technical terms this example might be for
            for term in technical_terms:
                # If the term appears closely before the example, it's likely an example of that term
                term_pos = text.lower().find(term.lower())
                if term_pos != -1:
                    example_pos = match.start()
                    # If term appears within 100 characters before the example
                    if 0 < example_pos - term_pos < 100:
                        terms_with_examples.add(term)
    
    # Print header for NER-based definitions
    print("\n===== NER-BASED DEFINITION DETECTION =====")
    
    # Use NER for more sophisticated analysis
    # Analyze sentence by sentence to find definitions and examples
    for sent_idx, sent in enumerate(doc.sents):
        sent_text = sent.text.lower()
        
        # Check for definitions in this sentence
        for term in technical_terms:
            term = term.lower()
            if term in sent_text:
                # Check for definition patterns in this specific sentence
                if (
                    re.search(r'is|are|refers to|means|defined as|called', sent_text) and
                    re.search(term, sent_text)
                ):
                    if term not in definition_matches:
                        definition_matches[term] = []
                    definition_matches[term].append(f"NER Sentence {sent_idx+1}")
                    print(f"Sentence {sent_idx+1} defines term '{term}'")
                    print(f"  Sentence: '{sent_text}'")
                    defined_terms.add(term)
        
        # Check for examples in this sentence
        for term in technical_terms:
            term = term.lower()
            if term in sent_text and (
                "example" in sent_text or 
                "e.g." in sent_text or 
                "such as" in sent_text or
                "like " in sent_text or
                "instance" in sent_text
            ):
                terms_with_examples.add(term)
    
    # Calculate the raw ratios
    if technical_terms:
        raw_defined_terms_ratio = len(defined_terms) / len(technical_terms)
        raw_example_coverage_ratio = len(terms_with_examples) / len(technical_terms)
    else:
        raw_defined_terms_ratio = 0.0
        raw_example_coverage_ratio = 0.0
    
    # Add more sophisticated NER-based term detection for definitions
    # Look for sentences that might be definitions but weren't caught by patterns
    technical_term_set = set(term.lower() for term in technical_terms)
    
    print("\n===== ADDITIONAL NER-BASED DEFINITION DETECTION =====")
    
    for sent_idx, sent in enumerate(doc.sents):
        # Check for sentences that look like definitions
        if len(sent) > 5 and len(sent) < 50:  # Definitions are usually medium-length sentences
            # Extract entities in this sentence
            entities = [ent.text.lower() for ent in sent.ents]
            
            # Check if any technical term appears as an entity or at the start of the sentence
            sent_start = sent.text.lower().split()[:3]  # First few words
            
            for term in technical_terms:
                term_lower = term.lower()
                # If term is an entity in this sentence, or appears at the start
                if (term_lower in entities or 
                    any(term_lower in start_word for start_word in sent_start)):
                    
                    # Check if sentence has verb patterns typical of definitions
                    has_definition_verb = False
                    verb_found = ""
                    for token in sent:
                        if token.pos_ == "VERB" and token.lemma_ in ["be", "mean", "refer", "define", "represent", "denote"]:
                            has_definition_verb = True
                            verb_found = token.text
                            break
                    
                    if has_definition_verb:
                        if term_lower not in defined_terms:
                            print(f"NER Entity/Start detected definition for '{term_lower}'")
                            print(f"  Sentence {sent_idx+1}: '{sent.text}'")
                            print(f"  Definition verb: '{verb_found}'")
                        
                        if term not in definition_matches:
                            definition_matches[term] = []
                        definition_matches[term].append(f"NER Entity/Start Sentence {sent_idx+1}")
                        defined_terms.add(term_lower)
    
    # For examples, use NLP to detect exemplification structures
    for sent_idx, sent in enumerate(doc.sents):
        sent_text = sent.text.lower()
        
        # Look for bullet points or numbered lists which often contain examples
        if re.match(r'^\s*[•\-\d]+\s+', sent_text):
            # Find which technical term this might be an example of
            for i in range(1, 6):  # Look up to 5 sentences back
                if i <= len(list(doc.sents)):
                    previous_sent = list(doc.sents)[-i].text.lower()
                    for term in technical_terms:
                        if term.lower() in previous_sent and (
                            "example" in previous_sent or 
                            "following" in previous_sent or
                            ":" in previous_sent
                        ):
                            terms_with_examples.add(term.lower())
    
    # Recalculate the raw ratios after the additional NER analysis
    if technical_terms:
        raw_defined_terms_ratio = len(defined_terms) / len(technical_terms)
        raw_example_coverage_ratio = len(terms_with_examples) / len(technical_terms)
    
    # Apply scaling based on what's reasonable in technical documentation
    defined_terms_ratio = scale_definition_coverage(raw_defined_terms_ratio, technical_terms, text)
    example_coverage_ratio = scale_example_coverage(raw_example_coverage_ratio, technical_terms, text)
    
    # Print summary of defined terms and how they were detected
    print("\n===== DEFINED TERMS SUMMARY =====")
    print(f"Found {len(defined_terms)} defined terms out of {len(technical_terms)} technical terms")
    print(f"Raw defined terms ratio: {raw_defined_terms_ratio:.4f}")
    print(f"Normalized defined terms ratio: {defined_terms_ratio:.4f}")
    
    print("\nDefined terms and detection methods:")
    for term in defined_terms:
        detection_methods = definition_matches.get(term, ["Unknown"])
        print(f"Term: '{term}' - Detected by: {', '.join(detection_methods)}")
    
    print("\n===== EXAMPLES SUMMARY =====")
    print(f"Found {len(terms_with_examples)} terms with examples out of {len(technical_terms)} technical terms")
    print(f"Raw examples ratio: {raw_example_coverage_ratio:.4f}")
    print(f"Normalized examples ratio: {example_coverage_ratio:.4f}")
    
    print("\nTerms with examples:")
    for term in terms_with_examples:
        print(f"- {term}")
    
    return defined_terms_ratio, example_coverage_ratio, list(defined_terms), list(terms_with_examples)

def scale_definition_coverage(raw_ratio, technical_terms, text):
    """
    Scale the definition coverage ratio based on what's reasonable in technical documentation.
    
    Not all technical terms need definitions in good technical documentation:
    1. Common terms in the field might not need definitions
    2. Some terms might be self-explanatory in context
    3. Different document types have different expectations
    
    Args:
        raw_ratio (float): Raw percentage of technical terms defined
        technical_terms (list): List of identified technical terms
        text (str): The full text being analyzed
        
    Returns:
        float: Scaled definition coverage ratio (0-1)
    """
    # Determine document type based on text features
    doc_length = len(text.split())
    has_abstract = bool(re.search(r'\babstract\b|summary', text[:500].lower()))
    has_references = bool(re.search(r'references|bibliography', text[-1000:].lower()))
    has_sections = len(re.findall(r'\n\s*#+\s+|\n\s*\d+\.\s+[A-Z]', text)) > 2
    
    # Make a guess at document type
    if has_abstract and has_references and doc_length > 2000:
        doc_type = "research_paper"
    elif has_sections and doc_length > 1000:
        doc_type = "technical_report"
    elif doc_length < 1000 and re.search(r'\bhow\s+to\b|\bguide\b|\btutorial\b', text.lower()):
        doc_type = "tutorial"
    else:
        doc_type = "general_technical"
    
    # Simple threshold-based scaling approach (similar to Gunning Fog)
    # Research papers
    if doc_type == "research_paper":
        if raw_ratio >= 0.6:  # Excellent definition coverage for research
            return 1.0
        elif raw_ratio >= 0.4:  # Good coverage
            return 0.8
        elif raw_ratio >= 0.25:  # Adequate coverage
            return 0.6
        elif raw_ratio >= 0.15:  # Minimal coverage
            return 0.4
        elif raw_ratio > 0:  # Poor coverage 
            return 0.2
        else:  # No coverage
            return 0.0
            
    # Tutorials (should have more definitions)
    elif doc_type == "tutorial":
        if raw_ratio >= 0.8:  # Excellent definition coverage for tutorials
            return 1.0
        elif raw_ratio >= 0.6:  # Good coverage
            return 0.8
        elif raw_ratio >= 0.4:  # Adequate coverage
            return 0.6
        elif raw_ratio >= 0.2:  # Minimal coverage
            return 0.4
        elif raw_ratio > 0:  # Poor coverage
            return 0.2
        else:  # No coverage
            return 0.0
            
    # Technical reports
    elif doc_type == "technical_report":
        if raw_ratio >= 0.7:  # Excellent definition coverage for tech reports
            return 1.0
        elif raw_ratio >= 0.5:  # Good coverage
            return 0.8
        elif raw_ratio >= 0.3:  # Adequate coverage
            return 0.6
        elif raw_ratio >= 0.15:  # Minimal coverage
            return 0.4
        elif raw_ratio > 0:  # Poor coverage
            return 0.2
        else:  # No coverage
            return 0.0
            
    # General technical documents
    else:  # general_technical
        if raw_ratio >= 0.65:  # Excellent definition coverage
            return 1.0
        elif raw_ratio >= 0.45:  # Good coverage
            return 0.8
        elif raw_ratio >= 0.25:  # Adequate coverage
            return 0.6
        elif raw_ratio >= 0.1:  # Minimal coverage
            return 0.4
        elif raw_ratio > 0:  # Poor coverage
            return 0.2
        else:  # No coverage
            return 0.0

def scale_example_coverage(raw_ratio, technical_terms, text):
    """
    Scale the example coverage ratio based on what's reasonable in technical documentation.
    
    Not all technical terms need examples in good technical documentation:
    1. Some concepts are clear without examples
    2. Different document types have different expectations for examples
    3. Some types of terms benefit more from examples than others
    
    Args:
        raw_ratio (float): Raw percentage of technical terms with examples
        technical_terms (list): List of identified technical terms
        text (str): The full text being analyzed
        
    Returns:
        float: Scaled example coverage ratio (0-1)
    """
    # Determine document type based on text features
    doc_length = len(text.split())
    has_abstract = bool(re.search(r'\babstract\b|summary', text[:500].lower()))
    has_references = bool(re.search(r'references|bibliography', text[-1000:].lower()))
    has_code_blocks = len(re.findall(r'```|    |\t', text)) > 5  # Check for code indentation or markdown blocks
    
    # Make a guess at document type
    if has_abstract and has_references and doc_length > 2000:
        doc_type = "research_paper"
    elif has_code_blocks and re.search(r'\bhow\s+to\b|\bguide\b|\btutorial\b', text.lower()):
        doc_type = "programming_tutorial"
    elif doc_length < 1500 and re.search(r'\bhow\s+to\b|\bguide\b|\btutorial\b', text.lower()):
        doc_type = "tutorial"
    else:
        doc_type = "general_technical"
    
    # Simple threshold-based scaling approach (similar to Gunning Fog)
    # Research papers (need fewer examples)
    if doc_type == "research_paper":
        if raw_ratio >= 0.4:  # Excellent example coverage for research
            return 1.0
        elif raw_ratio >= 0.25:  # Good coverage
            return 0.8
        elif raw_ratio >= 0.15:  # Adequate coverage
            return 0.6
        elif raw_ratio >= 0.05:  # Minimal examples
            return 0.4
        elif raw_ratio > 0:  # Very few examples
            return 0.2
        else:  # No examples
            return 0.0
            
    # Programming tutorials (need more examples)
    elif doc_type == "programming_tutorial":
        if raw_ratio >= 0.7:  # Excellent example coverage for programming tutorials
            return 1.0
        elif raw_ratio >= 0.5:  # Good coverage
            return 0.8
        elif raw_ratio >= 0.3:  # Adequate coverage
            return 0.6
        elif raw_ratio >= 0.15:  # Minimal examples
            return 0.4
        elif raw_ratio > 0:  # Very few examples
            return 0.2
        else:  # No examples
            return 0.0
            
    # General tutorials
    elif doc_type == "tutorial":
        if raw_ratio >= 0.6:  # Excellent example coverage for tutorials
            return 1.0
        elif raw_ratio >= 0.4:  # Good coverage
            return 0.8
        elif raw_ratio >= 0.25:  # Adequate coverage
            return 0.6
        elif raw_ratio >= 0.1:  # Minimal examples
            return 0.4
        elif raw_ratio > 0:  # Very few examples
            return 0.2
        else:  # No examples
            return 0.0
            
    # General technical documents
    else:  # general_technical
        if raw_ratio >= 0.5:  # Excellent example coverage
            return 1.0
        elif raw_ratio >= 0.35:  # Good coverage
            return 0.8
        elif raw_ratio >= 0.2:  # Adequate coverage
            return 0.6
        elif raw_ratio >= 0.08:  # Minimal examples
            return 0.4
        elif raw_ratio > 0:  # Very few examples
            return 0.2
        else:  # No examples
            return 0.0

def evaluate_definitions_and_examples_with_llm(text, technical_terms):
    """
    Use LLM to evaluate the percentage of technical terms that are defined
    and the coverage of examples in the text.
    
    This function is kept for backward compatibility but now calls the NER-based version.
    
    Args:
        text (str): The text to evaluate
        technical_terms (list): List of identified technical terms
        
    Returns:
        tuple: (defined_terms_ratio, example_coverage_ratio)
    """
    # First try the NER-based approach
    try:
        defined_terms_ratio, example_coverage_ratio, _, _ = evaluate_definitions_and_examples_with_ner(
            text, technical_terms
        )
        
        # If the NER approach found sufficient coverage, use those results
        if defined_terms_ratio > 0 or example_coverage_ratio > 0:
            return defined_terms_ratio, example_coverage_ratio
    except Exception as e:
        print(f"NER-based definition/example analysis failed: {e}. Falling back to LLM.")
    
    # Fallback to LLM approach if NER didn't find anything or failed
    # Create a prompt for the LLM
    technical_terms_list = ", ".join(technical_terms[:30])  # Limit to first 30 terms to avoid token limits
    
    prompt = f"""Analyze the following technical text and evaluate:
    
    1. What percentage of the technical terms are properly defined or explained in the text?
    2. What percentage of the key concepts have supporting examples?
    
    Technical terms identified in the text include: {technical_terms_list}
    
    Text to analyze:
    ```
    {text[:4000]}  # Limiting text length
    ```
    
    Please respond in JSON format:
    {{
        "defined_terms_percentage": <0.0-1.0>,  # The ratio of properly defined technical terms
        "examples_coverage_percentage": <0.0-1.0>,  # The ratio of key concepts with examples
        "defined_terms_analysis": "brief explanation of your analysis",
        "examples_analysis": "brief explanation of your analysis"
    }}
    """
    
    try:
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
            evaluation = json.loads(json_match.group(0))
            defined_terms_ratio = float(evaluation.get('defined_terms_percentage', 0.0))
            example_coverage_ratio = float(evaluation.get('examples_coverage_percentage', 0.0))
        else:
            print("Warning: Failed to parse LLM response for term definitions and examples")
            defined_terms_ratio = 0.0
            example_coverage_ratio = 0.0
            
    except Exception as e:
        print(f"Warning: LLM evaluation of definitions and examples failed: {str(e)}")
        defined_terms_ratio = 0.0
        example_coverage_ratio = 0.0
    
    return defined_terms_ratio, example_coverage_ratio

# Normalize Gunning Fog Index for technical content (target 12-14)
def normalize_gunning_fog(gunning_fog):
    if gunning_fog >= 18:  # Extremely complex, even for technical content
        return 0.2
    elif gunning_fog >= 16:  # Very complex technical content
        return 0.4
    elif gunning_fog >= 14:  # Upper end of optimal range
        return 0.8
    elif gunning_fog >= 12:  # Optimal range for technical content
        return 1.0
    elif gunning_fog >= 10:  # Slightly less complex than optimal
        return 0.8
    elif gunning_fog >= 8:  # Too simple for technical content
        return 0.6
    else:  # Far too simple for technical audience
        return 0.4

def calculate_structure(text):
    """
    Calculate structure metrics for a given text.
    
    Returns:
        dict: Dictionary containing structure metrics:
            - topic_hierarchy: Topic hierarchy score using LDA-based analysis (normalized to 0-1)
            - syntax_complexity: Language structure metrics based on dependency parsing
            - llm_evaluation: LLM-based structure evaluation
    """
    # Get topic hierarchy and syntax metrics
    analysis_results = estimate_concept_hierarchy_depth(text)
    topic_hierarchy_score = analysis_results.get('topic_hierarchy_score', 0.5)
    syntax_complexity_score = analysis_results.get('syntax_complexity_score', 0.5)
    
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
            0.4 * topic_hierarchy_score +  # 40% weight to topic hierarchy
            0.2 * syntax_complexity_score + # 20% weight to syntax complexity
            0.4 * llm_score                # 40% weight to LLM evaluation
        )
        if np.isnan(combined_score):
            combined_score = 0.5  # Default to neutral score if calculation fails
    except Exception as e:
        print(f"Warning: Error calculating combined structure score: {str(e)}")
        combined_score = 0.5
    
    return {
        'topic_hierarchy': topic_hierarchy_score,
        'syntax_complexity': syntax_complexity_score,
        'llm_evaluation': {
            'score': llm_score,
            'justification': llm_justification
        },
        'combined_score': float(combined_score),  # Ensure we return a float, not numpy float
        'weights': {
            'topic_hierarchy_weight': 0.4,
            'syntax_complexity_weight': 0.2,
            'llm_weight': 0.4
        }
    }

def calculate_final_weighted_score(text, llm_technical_depth=None, llm_clarity=None):
    # Calculate all component metrics
    technical_metrics = calculate_technical_depth(text)
    clarity_metrics = calculate_clarity(text)
    structure_metrics = calculate_structure(text)
    
    # Create consolidated metrics dictionary
    metrics = {
        'technical_term_count': technical_metrics['technical_term_metrics']['raw_count'],
        'gunning_fog_score': clarity_metrics['flesch_score'],
        'defined_terms_count': clarity_metrics['defined_terms_count'],
        'example_count': clarity_metrics['example_count'],
        'coherence': clarity_metrics['coherence'],
        'topic_hierarchy': structure_metrics['topic_hierarchy'],
        'syntax_complexity': structure_metrics['syntax_complexity'],
        'word_count': len(text.split()),
        'technical_term_metrics': technical_metrics['technical_term_metrics']
    }
    
    # Add LLM scores if available
    llm_results = {
        # Use the LLM evaluation from technical_metrics if available
        'technical_depth': technical_metrics.get('llm_evaluation', {'score': 0.5}),
        'clarity': clarity_metrics.get('llm_evaluation', {'score': 0.5}),
        'structure': structure_metrics.get('llm_evaluation', {'score': 0.5})
    }
    
    # Only override if explicitly provided
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
    Calculate a weighted score (0-1) based on various metrics with modified weights
    """
    # Extract the coverage density index related values
    unique_technical_terms = report_metrics.get('technical_term_metrics', {}).get('unique_terms', 0)
    if isinstance(unique_technical_terms, list):
        unique_technical_terms = len(unique_technical_terms)
    
    total_words = report_metrics.get('word_count', 0)
    
    # Calculate CDI with protection against division by zero
    cdi = unique_technical_terms / (total_words ** 0.5) if total_words > 0 else 0
    
    # Normalize CDI to 0-1 scale
    if cdi >= 1.5:  # Very high technical density
        cdi_score = 1.0
    elif cdi >= 1.0:
        cdi_score = 0.9
    elif cdi >= 0.8:
        cdi_score = 0.8
    elif cdi >= 0.6:
        cdi_score = 0.7
    elif cdi >= 0.4:
        cdi_score = 0.6
    elif cdi >= 0.3:
        cdi_score = 0.5
    elif cdi >= 0.2:
        cdi_score = 0.4
    elif cdi >= 0.1:
        cdi_score = 0.3
    elif cdi >= 0.05:
        cdi_score = 0.2
    else:
        cdi_score = 0.1
    
    # Get flow score from coherence
    flow_score = 0.5  # Default value
    if isinstance(report_metrics.get('coherence'), dict):
        flow_score = report_metrics['coherence'].get('concept_flow', {}).get('flow_score', 0.5)
    
    # Get topic hierarchy and syntax complexity values
    topic_hierarchy = report_metrics.get('topic_hierarchy', 0.5)
    syntax_complexity = report_metrics.get('syntax_complexity', 0.5)
    
    # Build scores dictionary
    scores = {
        # Technical Depth
        'technical_depth': {
            'cdi': cdi_score,
            'llm_evaluation': llm_results['technical_depth']['score']
        },
        
        # Clarity
        'clarity': {
            'gunning_fog': normalize_gunning_fog(report_metrics.get('gunning_fog_score', 12)),
            'defined_terms': report_metrics.get('defined_terms_count', 0.5),
            'examples': report_metrics.get('example_count', 0.5),
            'coherence': flow_score,
            'llm_evaluation': llm_results['clarity']['score']
        },
        
        # Structure
        'structure': {
            'topic_hierarchy': topic_hierarchy,
            'syntax_complexity': syntax_complexity,
            'llm_evaluation': llm_results['structure']['score']
        }
    }
    
    # Define component weights
    component_weights = {
        'technical_depth': 0.35,  # Technical depth weight
        'clarity': 0.35,          # Clarity weight
        'structure': 0.30         # Structure weight
    }
    
    # Define internal weights within each component
    internal_weights = {
        'technical_depth': {
            'cdi': 0.5,
            'llm_evaluation': 0.5
        },
        'clarity': {
            'gunning_fog': 0.15,
            'defined_terms': 0.15,
            'examples': 0.15,
            'coherence': 0.25,
            'llm_evaluation': 0.30
        },
        'structure': {
            'topic_hierarchy': 0.4,
            'syntax_complexity': 0.2,
            'llm_evaluation': 0.4
        }
    }
    
    # Calculate component scores
    component_scores = {}
    for component, metrics in scores.items():
        component_scores[component] = sum(
            value * internal_weights[component][metric]
            for metric, value in metrics.items()
        )
    
    # Calculate final weighted score
    final_score = sum(
        score * component_weights[component]
        for component, score in component_scores.items()
    )
    
    # Return detailed results
    return {
        'final_score': round(final_score, 3),
        'component_scores': {
            'technical_depth': round(component_scores['technical_depth'], 3),
            'clarity': round(component_scores['clarity'], 3),
            'structure': round(component_scores['structure'], 3)
        },
        'detailed_scores': {
            'technical_depth': {
                'cdi': round(scores['technical_depth']['cdi'], 3),
                'cdi_raw': round(cdi, 3),
                'llm_evaluation': round(scores['technical_depth']['llm_evaluation'], 3)
            },
            'clarity': {
                'gunning_fog': round(scores['clarity']['gunning_fog'], 3),
                'defined_terms': round(scores['clarity']['defined_terms'], 3),
                'examples': round(scores['clarity']['examples'], 3),
                'coherence': round(scores['clarity']['coherence'], 3),
                'llm_evaluation': round(scores['clarity']['llm_evaluation'], 3)
            },
            'structure': {
                'topic_hierarchy': round(scores['structure']['topic_hierarchy'], 3),
                'syntax_complexity': round(scores['structure']['syntax_complexity'], 3),
                'llm_evaluation': round(scores['structure']['llm_evaluation'], 3)
            }
        },
        'component_weights': component_weights,
        'internal_weights': internal_weights
    }

def compare_report_scores(original_metrics, improved_metrics, llm_results):
    """
    Compare the weighted scores between original and improved reports
    """
    original_score = calculate_weighted_score(original_metrics, llm_results['original'])
    improved_score = calculate_weighted_score(improved_metrics, llm_results['improved'])
    
    score_difference = improved_score['final_score'] - original_score['final_score']
    percent_improvement = (score_difference / original_score['final_score']) * 100 if original_score['final_score'] > 0 else 0
    
    # Compare component scores
    component_differences = {}
    for component in ['technical_depth', 'clarity', 'structure']:
        component_differences[component] = round(
            improved_score['component_scores'][component] - 
            original_score['component_scores'][component], 
            3
        )
    
    # Compare detailed scores for each component
    detailed_differences = {
        'technical_depth': {},
        'clarity': {},
        'structure': {}
    }
    
    # Technical depth detailed differences
    for metric in ['cdi', 'llm_evaluation']:
        detailed_differences['technical_depth'][metric] = round(
            improved_score['detailed_scores']['technical_depth'][metric] - 
            original_score['detailed_scores']['technical_depth'][metric],
            3
        )
    
    # Clarity detailed differences
    for metric in ['gunning_fog', 'defined_terms', 'examples', 'coherence', 'llm_evaluation']:
        detailed_differences['clarity'][metric] = round(
            improved_score['detailed_scores']['clarity'][metric] - 
            original_score['detailed_scores']['clarity'][metric],
            3
        )
    
    # Structure detailed differences
    for metric in ['topic_hierarchy', 'syntax_complexity', 'llm_evaluation']:
        detailed_differences['structure'][metric] = round(
            improved_score['detailed_scores']['structure'][metric] - 
            original_score['detailed_scores']['structure'][metric],
            3
        )
    
    return {
        'original': original_score,
        'improved': improved_score,
        'difference': round(score_difference, 3),
        'percent_improvement': round(percent_improvement, 2),
        'component_differences': component_differences,
        'detailed_differences': detailed_differences
    }

def create_weighted_scores_chart(weighted_scores, output_dir, timestamp):
    """
    Create a detailed chart showing the weighted scores comparison
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create first chart: Main component scores
    plt.figure(figsize=(15, 10))
    
    # Create data for the plot
    categories = ['Technical\nDepth', 'Clarity', 'Structure', 'Final Score']
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
    plt.title('Main Component Weighted Score Comparison')
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
    
    # Create detailed charts for each component
    component_names = ['technical_depth', 'clarity', 'structure']
    component_titles = ['Technical Depth', 'Clarity', 'Structure']
    
    for idx, component in enumerate(component_names):
        plt.figure(figsize=(15, 8))
        
        # Get metrics for this component
        metrics = list(weighted_scores['original']['detailed_scores'][component].keys())
        metric_labels = [m.replace('_', '\n').title() for m in metrics]
        
        original_values = [weighted_scores['original']['detailed_scores'][component][m] for m in metrics]
        improved_values = [weighted_scores['improved']['detailed_scores'][component][m] for m in metrics]
        
        # Set up the bar chart
        x = np.arange(len(metrics))
        
        # Create bars
        plt.bar(x - width/2, original_values, width, label='Original', color='#1f77b4', alpha=0.8)
        plt.bar(x + width/2, improved_values, width, label='Improved', color='#2ca02c', alpha=0.8)
        
        # Customize the plot
        plt.xlabel('Metrics')
        plt.ylabel('Score (0-1)')
        plt.title(f'{component_titles[idx]} Detailed Scores')
        plt.xticks(x, metric_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of each bar
        for i, v in enumerate(original_values):
            plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        for i, v in enumerate(improved_values):
            plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement arrows and labels
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
        
        # Add weights as a subtitle
        internal_weights = weighted_scores['improved']['internal_weights'][component]
        weight_text = "Weights: " + ", ".join([
            f"{metric.title()}: {weight*100:.0f}%" 
            for metric, weight in internal_weights.items()
        ])
        
        plt.figtext(
            0.5, 0.02,
            weight_text,
            ha='center',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
        
        # Set y-axis limit to 0-1 with some padding
        plt.ylim(0, 1.05)
        
        # Save the chart
        component_chart_path = os.path.join(output_dir, f"{component}_{timestamp}.png")
        plt.savefig(component_chart_path, dpi=300, bbox_inches='tight')
        print(f"Created {component} detailed chart: {component_chart_path}")
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
            original_report, improved_report, original_score, improved_score, original_coherence, improved_coherence, original_balanced_score, improved_balanced_score, original_technical_depth, improved_technical_depth, original_clarity, improved_clarity, original_structure, improved_structure = process_report(json_path)
            
            # Calculate report lengths
            original_length = len(original_report.split())
            improved_length = len(improved_report.split())
            
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
                    "gunning_fog_score": round(original_score, 2),
                    "actionable_recommendations_count": 0,
                    "technical_term_count": original_tech_metrics['raw_count'],
                    "dictionary_coverage_percentage": original_tech_metrics.get('dictionary_coverage_percentage', 0),
                    "balanced_technical_score": original_balanced_score,
                    "contextual_coherence": original_coherence,
                    "technical_depth": original_technical_depth,
                    "clarity": original_clarity,
                    "structure": original_structure
                },
                "improved_report": {
                    "word_count": improved_length,
                    "gunning_fog_score": round(improved_score, 2),
                    "actionable_recommendations_count": 0,
                    "technical_term_count": improved_tech_metrics['raw_count'],
                    "dictionary_coverage_percentage": improved_tech_metrics.get('dictionary_coverage_percentage', 0),
                    "balanced_technical_score": improved_balanced_score,
                    "contextual_coherence": improved_coherence,
                    "technical_depth": improved_technical_depth,
                    "clarity": improved_clarity,
                    "structure": improved_structure
                },
                "comparison": {
                    "word_count_difference": improved_length - original_length,
                    "word_count_percent_change": round(((improved_length - original_length) / original_length) * 100, 2),
                    "gunning_fog_difference": round(improved_score - original_score, 2),
                    "technical_terms_difference": improved_tech_metrics['raw_count'] - original_tech_metrics['raw_count'],
                    "dictionary_coverage_difference": improved_tech_metrics.get('dictionary_coverage_percentage', 0) - original_tech_metrics.get('dictionary_coverage_percentage', 0),
                    "balanced_technical_score_difference": improved_balanced_score - original_balanced_score,
                    "coherence_differences": {
                        "flow_score_change": improved_coherence['concept_flow']['flow_score'] - 
                                           original_coherence['concept_flow']['flow_score']
                    },
                    "technical_depth_difference": improved_technical_depth['combined_score'] - original_technical_depth['combined_score'],
                    "clarity_difference": improved_clarity['combined_score'] - original_clarity['combined_score'],
                    "structure_difference": improved_structure['combined_score'] - original_structure['combined_score']
                },
                "llm_evaluation": llm_results
            }
            
            # Calculate weighted scores
            weighted_scores = compare_report_scores(
                {
                    'technical_term_metrics': original_tech_metrics,
                    'word_count': original_length,
                    'gunning_fog_score': original_score,
                    'defined_terms_count': original_clarity.get('defined_terms_count', 0),
                    'example_count': original_clarity.get('example_count', 0),
                    'coherence': original_clarity.get('coherence', {}),
                    'topic_hierarchy': original_structure.get('topic_hierarchy', 0.5),
                    'syntax_complexity': original_structure.get('syntax_complexity', 0.5)
                },
                {
                    'technical_term_metrics': improved_tech_metrics,
                    'word_count': improved_length,
                    'gunning_fog_score': improved_score,
                    'defined_terms_count': improved_clarity.get('defined_terms_count', 0),
                    'example_count': improved_clarity.get('example_count', 0),
                    'coherence': improved_clarity.get('coherence', {}),
                    'topic_hierarchy': improved_structure.get('topic_hierarchy', 0.5),
                    'syntax_complexity': improved_structure.get('syntax_complexity', 0.5)
                },
                results['llm_evaluation']
            )

            # Add scores to results
            results['weighted_scores'] = weighted_scores

            # Print the simple report
            print("REPORT STATISTICS")
            print("=================")
            print(f"Original report word count: {original_length}")
            print(f"Improved report word count: {improved_length}")
            print(f"Original Gunning Fog Index: {original_score:.2f}")
            print(f"Improved Gunning Fog Index: {improved_score:.2f}")
            
            # Show change
            diff = improved_score - original_score
            direction = "higher" if diff > 0 else "lower" if diff < 0 else "unchanged"
            print(f"Readability change: {abs(diff):.2f} points {direction}")
            
            # Print technical metrics
            print("\nTECHNICAL METRICS")
            print("=================")
            print(f"Technical depth: {original_technical_depth['combined_score']:.2f} → {improved_technical_depth['combined_score']:.2f}")
            print(f"Technical terms: {original_tech_metrics['raw_count']} → {improved_tech_metrics['raw_count']}")
            print(f"Balanced technical score: {original_balanced_score:.2f}% → {improved_balanced_score:.2f}%")
            
            # Print clarity metrics
            print("\nCLARITY METRICS")
            print("==============")
            print(f"Overall clarity: {original_clarity['combined_score']:.2f} → {improved_clarity['combined_score']:.2f}")
            print(f"Defined terms: {original_clarity.get('defined_terms_count', 0):.2f} → {improved_clarity.get('defined_terms_count', 0):.2f}")
            print(f"Examples: {original_clarity.get('example_count', 0):.2f} → {improved_clarity.get('example_count', 0):.2f}")
            print(f"Coherence flow: {original_coherence['concept_flow']['flow_score']:.2f} → {improved_coherence['concept_flow']['flow_score']:.2f}")
            
            # Print structure metrics
            print("\nSTRUCTURE METRICS")
            print("================")
            print(f"Overall structure: {original_structure['combined_score']:.2f} → {improved_structure['combined_score']:.2f}")
            print(f"Topic hierarchy: {original_structure.get('topic_hierarchy', 0.5):.2f} → {improved_structure.get('topic_hierarchy', 0.5):.2f}")
            print(f"Syntax complexity: {original_structure.get('syntax_complexity', 0.5):.2f} → {improved_structure.get('syntax_complexity', 0.5):.2f}")
            
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
