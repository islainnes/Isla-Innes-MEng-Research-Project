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
    
    # Calculate Flesch scores
    original_score = textstat.flesch_reading_ease(original_report)
    improved_score = textstat.flesch_reading_ease(improved_report)
    
    # Initialize coherence analyzer
    coherence_analyzer = ContextualCoherenceAnalyzer()
    
    # Add coherence metrics
    original_coherence = coherence_analyzer.analyze_contextual_coherence(original_report)
    improved_coherence = coherence_analyzer.analyze_contextual_coherence(improved_report)
    
    return original_report, improved_report, original_score, improved_score, original_coherence, improved_coherence

def count_technical_terms_with_ner(text):
    """
    Count and identify technical terms using a comprehensive semiconductor dictionary
    first, then supplement with NLP techniques. Returns both raw counts and
    frequency normalized to 0-1 scale using predefined ranges based on document length.
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
    
    # Initialize container for all identified technical terms
    identified_terms = []
    total_words = len(text.split())
    
    # 1. Dictionary-based identification
    # Tokenize text for basic word-level matching
    words = re.findall(r'\b[a-zA-Z0-9][\w\-\.]*[a-zA-Z0-9]\b|\b[a-zA-Z0-9]\b', text.lower())
    
    # Check single words against dictionary
    for word in words:
        if word.lower() in semiconductor_dictionary:
            identified_terms.append(word.lower())
    
    # Check for multi-word terms from the dictionary
    for term in semiconductor_dictionary:
        if ' ' in term and term.lower() in text.lower():
            # Count each occurrence
            count = text.lower().count(term.lower())
            for _ in range(count):
                identified_terms.append(term.lower())
    
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
                    identified_terms.append(term)
        
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
                    identified_terms.append(term)
        
        # 2.3. Pattern matching for chemical formulas, measurements, etc.
        chemical_formula_pattern = r'\b[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)+\b'
        measurement_pattern = r'\b\d+(?:\.\d+)?(?:n|µ|m|k|M|G)?(?:m|A|V|W|Hz|eV|Ω|F|H)\b'
        
        chemical_formulas = re.findall(chemical_formula_pattern, text)
        for formula in chemical_formulas:
            if formula.lower() not in semiconductor_dictionary:
                identified_terms.append(formula.lower())
                
        measurements = re.findall(measurement_pattern, text)
        for measurement in measurements:
            # Don't add simple numbers as technical terms
            if any(unit in measurement for unit in ['m', 'A', 'V', 'W', 'Hz', 'eV', 'Ω', 'F', 'H']):
                identified_terms.append(measurement.lower())
        
    except Exception as e:
        print(f"Error in NLP-based technical term analysis: {str(e)}")
    
    # Calculate metrics
    raw_count = len(identified_terms)
    
    # Create term frequency dictionary
    term_frequencies = {}
    for term in identified_terms:
        if term in term_frequencies:
            term_frequencies[term] += 1
        else:
            term_frequencies[term] = 1
    
    # Get unique terms
    unique_terms = set(identified_terms)
    unique_count = len(unique_terms)
    
    # Calculate technical frequency using raw count (not unique terms)
    technical_frequency = raw_count / max(1, total_words)
    technical_percentage = technical_frequency * 100  # Convert to percentage for readability
    
    # Calculate percentage of dictionary covered
    dictionary_terms_found = set()
    for term in unique_terms:
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
    
    # Log the normalization details for debugging
    normalization_details = {
        'document_length': total_words,
        'document_category': doc_length_category,
        'technical_term_count': raw_count,
        'technical_frequency': technical_frequency,
        'technical_percentage': technical_percentage,
        'normalized_score': normalized_score,
        'dictionary_coverage_percentage': dictionary_coverage_percentage
    }
    
    return {
        'raw_count': raw_count,
        'unique_terms': unique_count,
        'term_frequencies': term_frequencies,
        'technical_frequency': technical_frequency,
        'normalized_score': normalized_score,
        'total_words': total_words,
        'identified_terms': list(unique_terms),  # Include the actual terms for transparency
        'dictionary_coverage_percentage': dictionary_coverage_percentage,  # Added dictionary coverage percentage
        'dictionary_terms_found': list(dictionary_terms_found),  # List of dictionary terms found in the text
        'total_dictionary_terms': len(semiconductor_dictionary),  # Total number of terms in the dictionary
        'normalization_details': normalization_details  # Added for transparency in the normalization process
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
        'dictionary_coverage_percentage',  # Added dictionary coverage percentage
        'example_count', 
        'defined_terms_count'
    ]
    
    tech_labels = [
        'Concept Hierarchy\nDepth',
        'Actionable\nRecommendations', 
        'Technical\nTerms',
        'Dictionary\nCoverage (%)',  # Added dictionary coverage percentage label
        'Examples', 
        'Defined\nTerms'
    ]
    
    x = np.arange(len(tech_metrics))
    
    # Handle potential missing dictionary_coverage_percentage in existing results
    if 'dictionary_coverage_percentage' not in original:
        original['dictionary_coverage_percentage'] = 0
    if 'dictionary_coverage_percentage' not in improved:
        improved['dictionary_coverage_percentage'] = 0
    
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
    flesch_score_change = (results["comparison"]["flesch_score_difference"] / original["flesch_score"]) * 100 if original["flesch_score"] != 0 else 0
    
    # Technical metrics
    concept_depth_change = results["comparison"]["concept_depth_difference"]
    recommendations_change = results["comparison"]["recommendations_difference"]
    tech_terms_change = results["comparison"]["technical_terms_difference"]
    
    # Dictionary coverage change
    if "dictionary_coverage_percentage" in original and "dictionary_coverage_percentage" in improved:
        dictionary_coverage_change = improved["dictionary_coverage_percentage"] - original["dictionary_coverage_percentage"]
    else:
        dictionary_coverage_change = 0
    
    examples_change = results["comparison"]["examples_difference"]
    defined_terms_change = results["comparison"]["defined_terms_difference"]
    
    # Add all changes and labels
    percent_changes = [
        word_count_change, 
        flesch_score_change, 
        concept_depth_change,
        recommendations_change, 
        tech_terms_change,
        dictionary_coverage_change,  # Added dictionary coverage change
        examples_change, 
        defined_terms_change
    ]
    
    metric_names = [
        'Word Count', 
        'Flesch Score', 
        'Concept Depth',
        'Recommendations', 
        'Technical Terms',
        'Dictionary Coverage',  # Added dictionary coverage label
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
    Calculate technical depth metrics using NER-based technical term analysis and LLM evaluation
    
    Returns a comprehensive dictionary of metrics including:
    - Technical term metrics (dictionary coverage only)
    - Concept hierarchy metrics (topic hierarchy, syntax complexity)
    - LLM evaluation
    - Combined score
    """
    # Get technical metrics with NER-based frequency analysis
    tech_metrics = count_technical_terms_with_ner(text)
    
    # Get concept hierarchy depth with split metrics
    concept_hierarchy = estimate_concept_hierarchy_depth(text)
    # Use the split metrics
    topic_hierarchy_score = concept_hierarchy['topic_hierarchy_score']
    syntax_complexity_score = concept_hierarchy['syntax_complexity_score'] 
    
    # Initialize result dictionary
    result = {
        'technical_term_metrics': tech_metrics,
        'concept_hierarchy_depth': concept_hierarchy
    }
    
    # Add LLM evaluation
    try:
        prompt = f"""Rigorously evaluate the technical depth of the following text with a highly critical eye. Consider:

1. TECHNICAL SOPHISTICATION
   - Does the text demonstrate advanced understanding of specialized concepts?
   - Are complex technical phenomena explained with precision and detail?
   - Is domain expertise evident or merely superficial?

2. CONCEPTUAL RIGOR
   - Are explanations technically complete and substantive?
   - Is superficial treatment of concepts avoided?
   - Does the text go beyond basic definitions to explain mechanisms and theory?

3. TERMINOLOGY AND PRECISION
   - Is specialized terminology used accurately and appropriately?
   - Are technical terms defined correctly with proper context?
   - Does the text maintain technical precision throughout?

4. DEPTH OF TECHNICAL ANALYSIS
   - Does the text explore concepts at a deep level rather than just mentioning them?
   - Are technical relationships, dependencies, and implications addressed?
   - Is there evidence of technical insight rather than just description?

5. TECHNICAL ACCESSIBILITY
   - Are complex technical terms clearly explained for the target audience?
   - Does the text provide appropriate context to make technical concepts understandable?
   - Are analogies, examples, or illustrations used to clarify difficult concepts?
   - Does the text balance technical depth with accessibility?

Be extremely strict in your evaluation. A high score should only be given to texts that demonstrate genuinely advanced technical knowledge and deep understanding.
        
        Text to evaluate:
        ```
        {text[:6000]}  # Limiting to first 6000 chars to keep within context window
        ```

Rate the technical depth on a scale of 0.0-1.0 where:
0.0-0.2: Superficial/introductory level with minimal technical content
0.3-0.4: Basic technical content with limited depth
0.5-0.6: Moderate technical depth suitable for informed practitioners
0.7-0.8: Advanced technical content with substantial depth
0.9-1.0: Expert-level technical content with exceptional depth and rigor
        
        Format your response as a JSON object with this structure:
        {{
            "score": <0.0-1.0>,
    "justification": "detailed explanation of strengths and weaknesses"
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
    
    # Get dictionary coverage score
    dictionary_coverage_score = min(tech_metrics.get('dictionary_coverage_percentage', 0) / 20.0, 1.0)
    
    # Calculate combined score with new weighting scheme (removed technical term score):
    # - Dictionary coverage: 25%
    # - Topic hierarchy: 25%
    # - Syntax complexity: 15% (now measures tree depth and complex clauses only)
    # - LLM evaluation: 35%
    result['combined_score'] = (
        0.25 * dictionary_coverage_score +  
        0.25 * topic_hierarchy_score +      
        0.15 * syntax_complexity_score +    
        0.35 * result['llm_evaluation']['score']
    )
    
    # Include details about the weighting in the result for transparency
    result['weighting_details'] = {
        'dictionary_coverage_weight': 0.25,
        'topic_hierarchy_weight': 0.25,
        'syntax_complexity_weight': 0.15,
        'llm_evaluation_weight': 0.35,
        'explanation': "Dictionary coverage assesses technical vocabulary breadth, topic hierarchy measures conceptual organization, syntax complexity evaluates sentence depth and complex clauses, and LLM evaluation provides human-like judgment."
    }
    
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
            
    Raises:
        Exception: If any part of the analysis fails
    """
    # Calculate basic metrics
    flesch_score = textstat.flesch_reading_ease(text)
    
    # Get technical term metrics to use as a base for determining how many terms should be defined
    tech_metrics = count_technical_terms_with_ner(text)
    unique_technical_terms = tech_metrics['unique_terms']
    technical_terms = tech_metrics['identified_terms']
    
    # Enhanced linguistic analysis for definitions and examples - no fallbacks
    import spacy
    try:
        nlp = spacy.load('en_core_web_lg')
    except OSError:
        nlp = spacy.load('en_core_web_sm')
    
    # Process the document
    doc = nlp(text)
    
    # Detect definitions using enhanced linguistic approach
    definitions, defined_concepts = detect_definitions_enhanced(doc, technical_terms)
    defined_terms_count = len(definitions)
    
    # Detect examples using enhanced approach
    examples = detect_examples_enhanced(doc)
    example_count = len(examples)
    
    # Calculate topic distribution
    if len(list(doc.sents)) >= 3:
        # Get sentence texts
        sentences = [sent.text for sent in doc.sents]
        
        # Use topic modeling to assign topics to sentences
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(sentences)
        
        num_topics = min(5, max(2, len(sentences) // 5))
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        topic_distributions = lda.fit_transform(doc_term_matrix)
        
        # Assign topics to sentences and examples
        sent_topics = [dist.argmax() for dist in topic_distributions]
        
        # Update examples with topic info
        examples_by_topic = {i: 0 for i in range(num_topics)}
        for i, sent in enumerate(doc.sents):
            if i < len(sent_topics):
                for ex in examples:
                    if ex["example"] == sent.text:
                        ex["topic"] = sent_topics[i]
                        examples_by_topic[sent_topics[i]] += 1
        
        # Calculate topic coverage by examples
        topics_with_examples = sum(1 for count in examples_by_topic.values() if count > 0)
        topic_coverage = topics_with_examples / max(1, num_topics)
    else:
        # Not enough content for meaningful topic analysis
        topic_coverage = 1.0 if example_count > 0 else 0.0
    
    # Check how many technical terms were defined
    if technical_terms:
        defined_technical_terms = sum(1 for term in technical_terms 
                                   if any(term in def_item["term"] or def_item["term"] in term 
                                       for def_item in definitions))
        technical_term_coverage = defined_technical_terms / max(1, len(technical_terms))
    else:
        technical_term_coverage = 1.0 if defined_terms_count > 0 else 0.0
    
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
    
    # Normalize definitions based on enhanced analysis
    if technical_term_coverage >= 0.7:  # Excellent coverage
        normalized_defined = 1.0
    elif technical_term_coverage >= 0.5:  # Good coverage
        normalized_defined = 0.8
    elif technical_term_coverage >= 0.3:  # Adequate coverage
        normalized_defined = 0.6
    elif technical_term_coverage > 0:  # Some coverage
        normalized_defined = 0.4
    else:  # No coverage
        normalized_defined = 0.1
    
    # Normalize examples based on technical terms rather than estimated concepts
    # Instead of this:
    # sentences = text.split('.')
    # total_sentences = len([s for s in sentences if len(s.strip()) > 10])
    # estimated_concepts = max(1, total_sentences // 3)
    # example_ratio = example_count / max(1, estimated_concepts)
    
    # Use technical terms as basis for normalization:
    example_ratio = example_count / max(1, unique_technical_terms * 0.5)  # Expect examples for ~50% of technical terms
    
    # Combine raw count with topic coverage
    if example_ratio >= 0.5 and topic_coverage >= 0.7:  # Excellent - examples for half of technical terms with good distribution
        normalized_examples = 1.0
    elif example_ratio >= 0.3 and topic_coverage >= 0.5:  # Good
        normalized_examples = 0.8
    elif example_ratio >= 0.2 and topic_coverage >= 0.3:  # Adequate
        normalized_examples = 0.6
    elif example_ratio > 0 or topic_coverage > 0:  # Some examples
        normalized_examples = 0.4
    else:  # No examples
        normalized_examples = 0.1
    
    # Add LLM evaluation - allow this one to fail independently
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
    
    # Calculate combined score with updated weights
    combined_score = (
        0.20 * normalized_flesch +      # 20% weight to readability
        0.20 * normalized_defined +     # 20% weight to defined terms
        0.20 * normalized_examples +    # 20% weight to examples
        0.40 * llm_score                # 40% weight to LLM evaluation
    )
    
    # Return dictionary with all metrics
    return {
        'flesch_score': normalized_flesch,
        'defined_terms_count': normalized_defined,
        'example_count': normalized_examples,
        'definition_coverage': technical_term_coverage,
        'example_topic_coverage': topic_coverage,
        'definition_details': definitions,
        'example_details': examples,
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
    
    Weights are grouped into main categories:
    1. Technical Vocabulary (25% total)
       - Dictionary Coverage (25%)
    2. Conceptual Organization (40% total)
       - Topic Hierarchy (25%)
       - Syntax Complexity (15%)
    3. LLM Evaluation (35%)
    """
    
    # Make sure we have the split concept hierarchy metrics
    if isinstance(report_metrics['concept_hierarchy_depth'], dict) and 'topic_hierarchy_score' in report_metrics['concept_hierarchy_depth']:
        # Split metrics are available
        topic_hierarchy = report_metrics['concept_hierarchy_depth']['topic_hierarchy_score']
        syntax_complexity = report_metrics['concept_hierarchy_depth']['syntax_complexity_score']
    else:
        # If metrics aren't split yet, use the existing combined score
        topic_hierarchy = report_metrics['concept_hierarchy_depth'] * 0.6  # Approximate based on original weighting
        syntax_complexity = report_metrics['concept_hierarchy_depth'] * 0.4  # Approximate based on original weighting
    
    # Calculate normalized scores (0-1 scale)
    scores = {
        # Technical Vocabulary
        'dictionary_coverage': min(report_metrics.get('dictionary_coverage_percentage', 0) / 20.0, 1.0),  # 20% coverage is max score
        
        # Conceptual Organization
        'topic_hierarchy': topic_hierarchy,
        'syntax_complexity': syntax_complexity,
        
        # LLM Evaluation
        'llm_evaluation': llm_results['technical_depth']['score']
    }
    
    # Define weights
    weights = {
        'dictionary_coverage': 0.25,    # 25%
        'topic_hierarchy': 0.25,        # 25%
        'syntax_complexity': 0.15,      # 15%
        'llm_evaluation': 0.35          # 35%
    }
    
    # Calculate the weighted score
    weighted_score = sum(scores[metric] * weight for metric, weight in weights.items())
    
    # Calculate the technical vocabulary score (now just dictionary coverage)
    technical_vocab_score = scores['dictionary_coverage']
    
    # Calculate the conceptual organization score
    conceptual_org_score = (scores['topic_hierarchy'] * weights['topic_hierarchy'] + 
                          scores['syntax_complexity'] * weights['syntax_complexity']) / 0.40
    
    return {
        'final_score': round(weighted_score, 3),
        'component_scores': {
            'technical_vocabulary': round(technical_vocab_score, 3),
            'conceptual_organization': round(conceptual_org_score, 3),
            'llm_evaluation': round(scores['llm_evaluation'], 3)
        },
        'detailed_scores': {
            'dictionary_coverage': round(scores['dictionary_coverage'], 3),
            'topic_hierarchy': round(scores['topic_hierarchy'], 3),
            'syntax_complexity': round(scores['syntax_complexity'], 3),
            'llm_evaluation': round(scores['llm_evaluation'], 3)
        },
        'weights': weights
    }

def compare_report_scores(original_metrics, improved_metrics, llm_results):
    """
    Compare the weighted scores between original and improved reports
    """
    original_score = calculate_weighted_score(original_metrics, llm_results['original'])
    improved_score = calculate_weighted_score(improved_metrics, llm_results['improved'])
    
    score_difference = improved_score['final_score'] - original_score['final_score']
    percent_improvement = (score_difference / original_score['final_score']) * 100 if original_score['final_score'] > 0 else 0
    
    return {
        'original': original_score,
        'improved': improved_score,
        'difference': round(score_difference, 2),
        'percent_improvement': round(percent_improvement, 2),
        'component_differences': {
            'technical_vocabulary': round(improved_score['component_scores']['technical_vocabulary'] - 
                                    original_score['component_scores']['technical_vocabulary'], 2),
            'conceptual_organization': round(improved_score['component_scores']['conceptual_organization'] - 
                            original_score['component_scores']['conceptual_organization'], 2),
            'llm_evaluation': round(improved_score['component_scores']['llm_evaluation'] - 
                              original_score['component_scores']['llm_evaluation'], 2)
        },
        'detailed_differences': {
            'dictionary_coverage': round(improved_score['detailed_scores']['dictionary_coverage'] - 
                                       original_score['detailed_scores']['dictionary_coverage'], 2),
            'topic_hierarchy': round(improved_score['detailed_scores']['topic_hierarchy'] - 
                                   original_score['detailed_scores']['topic_hierarchy'], 2),
            'syntax_complexity': round(improved_score['detailed_scores']['syntax_complexity'] - 
                                     original_score['detailed_scores']['syntax_complexity'], 2),
            'llm_evaluation': round(improved_score['detailed_scores']['llm_evaluation'] - 
                                  original_score['detailed_scores']['llm_evaluation'], 2)
        }
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
    categories = ['Technical\nVocabulary', 'Conceptual\nOrganization', 'LLM\nEvaluation', 'Final Score']
    original_values = [
        weighted_scores['original']['component_scores']['technical_vocabulary'],
        weighted_scores['original']['component_scores']['conceptual_organization'],
        weighted_scores['original']['component_scores']['llm_evaluation'],
        weighted_scores['original']['final_score']
    ]
    improved_values = [
        weighted_scores['improved']['component_scores']['technical_vocabulary'],
        weighted_scores['improved']['component_scores']['conceptual_organization'],
        weighted_scores['improved']['component_scores']['llm_evaluation'],
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
    
    # Create second chart: Detailed component scores
    plt.figure(figsize=(15, 10))
    
    # Create data for the plot
    detailed_categories = [
        'Dictionary\nCoverage', 
        'Topic\nHierarchy', 
        'Syntax\nComplexity', 
        'LLM\nEvaluation'
    ]
    
    detailed_original_values = [
        weighted_scores['original']['detailed_scores']['dictionary_coverage'],
        weighted_scores['original']['detailed_scores']['topic_hierarchy'],
        weighted_scores['original']['detailed_scores']['syntax_complexity'],
        weighted_scores['original']['detailed_scores']['llm_evaluation']
    ]
    
    detailed_improved_values = [
        weighted_scores['improved']['detailed_scores']['dictionary_coverage'],
        weighted_scores['improved']['detailed_scores']['topic_hierarchy'],
        weighted_scores['improved']['detailed_scores']['syntax_complexity'],
        weighted_scores['improved']['detailed_scores']['llm_evaluation']
    ]
    
    # Calculate component differences
    detailed_differences = [
        weighted_scores['detailed_differences']['dictionary_coverage'],
        weighted_scores['detailed_differences']['topic_hierarchy'],
        weighted_scores['detailed_differences']['syntax_complexity'],
        weighted_scores['detailed_differences']['llm_evaluation']
    ]
    
    # Set up the bar chart
    x = np.arange(len(detailed_categories))
    
    # Create bars
    plt.bar(x - width/2, detailed_original_values, width, label='Original', color='#1f77b4', alpha=0.8)
    plt.bar(x + width/2, detailed_improved_values, width, label='Improved', color='#2ca02c', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Detailed Components')
    plt.ylabel('Score (0-1)')
    plt.title('Detailed Component Weighted Score Comparison')
    plt.xticks(x, detailed_categories)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(detailed_original_values):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(detailed_improved_values):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement arrows and labels
    for i in range(len(detailed_categories)):
        diff = detailed_differences[i]
        if diff != 0:
            # Arrow color based on difference
            arrow_color = 'green' if diff > 0 else 'red'
            
            # Position arrow between the bars
            arrow_x = i
            arrow_y_start = min(detailed_original_values[i], detailed_improved_values[i]) + (abs(diff) / 2)
            
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
    weights = weighted_scores['improved']['weights']
    weight_text = (
        f"Weights: Dictionary: {weights['dictionary_coverage']*100:.0f}%, "
        f"Topic: {weights['topic_hierarchy']*100:.0f}%, "
        f"Syntax: {weights['syntax_complexity']*100:.0f}%, "
        f"LLM: {weights['llm_evaluation']*100:.0f}%"
    )
    
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
    
    # Save the detailed chart
    detailed_chart_path = os.path.join(output_dir, f"detailed_weighted_scores_{timestamp}.png")
    plt.savefig(detailed_chart_path, dpi=300, bbox_inches='tight')
    print(f"Created detailed weighted scores chart: {detailed_chart_path}")
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

def detect_definitions_enhanced(doc, technical_terms=None):
    """
    Enhanced detection of definitions using dependency parsing.
    
    Args:
        doc: spaCy processed document
        technical_terms: List of technical terms to check for definitions
        
    Returns:
        List of dictionaries with terms and their definitions
    """
    definitions = []
    defined_concepts = set()
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        
        # Check if the sentence contains definitional verbs or constructs
        is_definition = False
        defined_term = None
        
        # Definitional verbs and patterns
        def_verbs = ["define", "mean", "refer", "be", "represent", "constitute", "denote", "signify"]
        def_patterns = ["is defined as", "refers to", "means", "is a", "is an", "is the", "can be defined"]
        
        # Look for definition patterns with dependency parsing
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break
        
        if root and root.lemma_ in def_verbs:
            # Find subject (potential defined term)
            subjects = [child for child in root.children if child.dep_ in ("nsubj", "nsubjpass")]
            
            # Find objects (potential definitions)
            objects = [child for child in root.children if child.dep_ in ("attr", "dobj", "pobj")]
            
            if subjects and objects:
                for subj in subjects:
                    # Extract complete noun phrase
                    term_tokens = [subj] + list(subj.children)
                    term_tokens.sort(key=lambda x: x.i)
                    defined_term = " ".join([t.text for t in term_tokens]).lower()
                    is_definition = True
        
        # Also check for explicit definition markers
        if any(pattern in sent_text.lower() for pattern in def_patterns):
            # Simple pattern matching for cases dependency parsing might miss
            for pattern in def_patterns:
                if pattern in sent_text.lower():
                    parts = sent_text.lower().split(pattern, 1)
                    if len(parts) == 2 and parts[0].strip():
                        defined_term = parts[0].strip()
                        is_definition = True
                        break
        
        if is_definition and defined_term:
            definitions.append({"term": defined_term, "definition": sent_text})
            defined_concepts.add(defined_term)
    
    return definitions, defined_concepts

def detect_examples_enhanced(doc):
    """
    Enhanced detection of examples using linguistic analysis.
    
    Args:
        doc: spaCy processed document
        
    Returns:
        List of dictionaries with examples and their details
    """
    examples = []
    
    # Look for example markers
    example_markers = ["for example", "for instance", "such as", "e.g.", "to illustrate", 
                      "specifically", "in particular", "namely", "like", "as seen in"]
    
    for i, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        
        is_example = False
        
        # Check for example markers
        if any(marker in sent_text.lower() for marker in example_markers):
            is_example = True
        
        # Check for enumerations that might be examples
        if not is_example and re.search(r'(?:^|\s)(?:1[.)]|first[,:])', sent_text.lower()):
            # Check if previous sentence has example introducer
            prev_sent = list(doc.sents)[i-1] if i > 0 else None
            if prev_sent and any(marker in prev_sent.text.lower() for marker in example_markers):
                is_example = True
        
        if is_example:
            examples.append({"example": sent_text})
    
    return examples

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
            original_tech_metrics = count_technical_terms_with_ner(original_report)
            original_examples = count_examples(original_report)
            original_defined_terms = count_defined_terms(original_report)
            
            # Calculate technical metrics for improved report
            improved_concept_depth = estimate_concept_hierarchy_depth(improved_report)
            improved_tech_metrics = count_technical_terms_with_ner(improved_report)
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
                    "technical_term_count": original_tech_metrics['raw_count'],
                    "dictionary_coverage_percentage": original_tech_metrics.get('dictionary_coverage_percentage', 0),  # Added dictionary coverage percentage
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
                    "technical_term_count": improved_tech_metrics['raw_count'],
                    "dictionary_coverage_percentage": improved_tech_metrics.get('dictionary_coverage_percentage', 0),  # Added dictionary coverage percentage
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
                    "technical_terms_difference": improved_tech_metrics['raw_count'] - original_tech_metrics['raw_count'],
                    "dictionary_coverage_difference": improved_tech_metrics.get('dictionary_coverage_percentage', 0) - original_tech_metrics.get('dictionary_coverage_percentage', 0),  # Added dictionary coverage difference
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
            print(f"Technical terms: {original_tech_metrics['raw_count']} → {improved_tech_metrics['raw_count']}")
            print(f"Dictionary coverage: {original_tech_metrics.get('dictionary_coverage_percentage', 0):.2f}% → {improved_tech_metrics.get('dictionary_coverage_percentage', 0):.2f}%")  # Added dictionary coverage
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
            print(f"Technical Depth: {weighted_scores['component_differences']['technical_vocabulary']}")
            print(f"Clarity: {weighted_scores['component_differences']['conceptual_organization']}")
            print(f"Structure: {weighted_scores['component_differences']['llm_evaluation']}")
            
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
