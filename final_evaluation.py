import re
import spacy
import numpy as np
import openai
import json
import textstat  # Add this import for Gunning Fog
from typing import Dict
import os  # Add this for environment variable access
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from dotenv import load_dotenv  # Add this import for reading .env files

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("OPENAI_API_KEY", "")  # Get API key from environment or empty string as fallback
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable not set")
openai_client = openai.OpenAI(api_key=api_key)

def analyze_sentence_complexity_normalized(text, nlp):
    """Analyzes syntactic complexity through dependency parsing, normalized to 0-1"""
    try:
        doc = nlp(text)
        
        # Calculate metrics
        # Tree depth: maximum dependency tree depth across sentences
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
        
        # More demanding normalization with finer granularity
        # Optimal range is now narrower (6-7) and harder to achieve
        if avg_max_depth < 3:
            normalized_depth = 0.2  # Too simple
        elif avg_max_depth < 4:
            normalized_depth = 0.4  # Somewhat simple
        elif avg_max_depth < 5:
            normalized_depth = 0.6  # Moderate complexity
        elif avg_max_depth < 6:
            normalized_depth = 0.8  # Good complexity
        elif avg_max_depth <= 7:
            normalized_depth = 1.0  # Optimal complexity
        elif avg_max_depth <= 8:
            normalized_depth = 0.8  # Slightly too complex
        elif avg_max_depth <= 9:
            normalized_depth = 0.6  # Too complex
        else:
            normalized_depth = 0.4  # Excessively complex
        
        # Return normalized depth as the final score (100% weighting)
        return normalized_depth
        
    except Exception as e:
        print(f"Error in syntax analysis: {e}")
        return 0.4  # Default fallback score

def evaluate_technical_depth_with_llm(text):
    """
    Use LLM to evaluate the technical depth of the text.
    
    Returns:
        float: Technical depth score (0-1)
    """
    try:
        # More critical prompt with explicit scoring guidance that discourages high scores unless truly deserved.
        prompt = f"""Rigorously evaluate the technical depth of the following text with an EXTREMELY critical eye. Be harsh in your assessment and avoid giving high scores unless truly deserved.

Consider:
        1. Technical vocabulary and terminology usage (Is it using appropriate field-specific terminology accurately?)
        2. Concept depth and complexity (Does it go beyond surface-level explanations?)
        3. Technical accuracy (Are technical details correct and precisely described?)
        4. Sophistication of analysis (Does it show deep understanding or merely superficial knowledge?)
        5. Use of advanced concepts (Does it incorporate cutting-edge or complex ideas?)
        
        Text to evaluate:
        ```
        {text[:6000]}  # Limiting to first 6000 chars to keep within context window
        ```
        
        Scoring guidance:
        - 0.9-1.0: ONLY for text with exceptional technical depth, equivalent to expert-level scientific publications
        - 0.7-0.8: Strong technical depth with advanced concepts, but not at publishable research level
        - 0.5-0.6: Moderate technical depth, appropriate for educated professionals but not specialists
        - 0.3-0.4: Basic technical content with some field-specific terminology
        - 0.0-0.2: Minimal technical content, mainly general knowledge
        
        BE VERY STRICT AND CRITICAL IN YOUR EVALUATION. Default to lower scores when in doubt. Only award high scores (>0.7) for truly exceptional technical depth.
        
        Format your response as a JSON object with this structure:
        {{
            "score": <0.0-1.0>,
            "justification": "brief explanation with specific examples of technical content quality"
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
        json_match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content)
        if json_match:
            llm_evaluation = json.loads(json_match.group(0))
            llm_evaluation_score = float(llm_evaluation.get('score', 0.5))
            llm_justification = llm_evaluation.get('justification', "No justification provided")
            return {
                'score': llm_evaluation_score,
                'justification': llm_justification
            }
        else:
            return {
                'score': 0.5,
                'justification': "Failed to parse LLM response"
            }
    except Exception as e:
        print(f"Warning: LLM evaluation failed: {str(e)}")
        return {
            'score': 0.5,
            'justification': f"LLM evaluation failed: {str(e)}"
        }

def calculate_technical_depth(text):
    """
    Count and identify technical terms using a comprehensive semiconductor dictionary
    first, then supplement with NLP techniques. Returns a balanced technical score
    based on dictionary terms, NER terms, syntax complexity, and LLM evaluation.
    """
    # Comprehensive dictionary of semiconductor and electronics terminology
    semiconductor_dictionary = {
        # Original Materials
        "silicon", "germanium", "gallium arsenide", "gaas", "gan", "sic", "silicon carbide",
        "semiconductor", "dielectric", "oxide", "nitride", "polysilicon", "silicon dioxide", "sio2",
        "silicon nitride", "si3n4", "high-k", "low-k", "copper", "aluminum", "tungsten", "titanium",
        "tantalum", "cobalt", "silicide", "germanium", "iii-v", "ii-vi", "compound semiconductor",
        "heterojunction", "quantum well", "quantum dot", "superlattice", "nanowire", "graphene", 
        "2d materials", "perovskite", "organic semiconductor",
        
        # Additional 2D Materials and Quantum Materials
        "monolayer", "bilayer", "few-layer", "van der waals", "transition metal dichalcogenide", "tmd",
        "molybdenum disulfide", "mos2", "tungsten diselenide", "wse2", "topological insulator",
        "weyl semimetal", "dirac semimetal", "boron nitride", "hbn", "hexagonal boron nitride",
        "phosphorene", "black phosphorus", "silicene", "germanene", "stanene", "mxene",
        "metal organic framework", "mof", "covalent organic framework", "cof",
        
        # Quantum Physics Terms
        "quantum confinement", "exciton", "trion", "biexciton", "polaron", "polariton",
        "quantum interference", "coherence", "decoherence", "wavefunction", "superposition",
        "entanglement", "quantum state", "fermi level", "density of states", "band structure",
        "brillouin zone", "spin-orbit coupling", "zeeman effect", "quantum hall effect",
        "integer quantum hall", "fractional quantum hall", "valley polarization", "valley degeneracy",
        "quantum phase transition", "coulomb blockade", "kondo effect", "casimir effect",
        "quantum critical point", "quantum fluctuation", "quantum oscillation",
        
        # Optoelectronics and Photonics
        "photoluminescence", "electroluminescence", "photocurrent", "photoresponse", "photodetector",
        "plasmon", "plasmonics", "surface plasmon", "localized surface plasmon", "waveguide",
        "photonic crystal", "metamaterial", "metasurface", "photonic integrated circuit",
        "optical cavity", "microcavity", "resonator", "quality factor", "finesse", "free spectral range",
        "group velocity", "phase velocity", "dispersion", "nonlinear optics", "second harmonic generation",
        "third harmonic generation", "optical parametric oscillation", "stimulated emission",
        "absorption", "reflection", "transmission", "scattering", "raman scattering",
        "brillouin scattering", "photonic bandgap", "optical gain", "laser", "maser",
        
        # Energy-related Terms
        "photovoltaic", "solar cell", "heterojunction solar cell", "bulk heterojunction",
        "perovskite solar cell", "tandem solar cell", "quantum dot solar cell", "battery",
        "lithium ion", "sodium ion", "fuel cell", "supercapacitor", "thermoelectric",
        "seebeck effect", "peltier effect", "thomson effect", "energy harvesting",
        "power conversion efficiency", "fill factor", "open-circuit voltage", "short-circuit current",
        
        # Characterization Methods
        "photoemission spectroscopy", "angle-resolved photoemission", "arpes", 
        "scanning tunneling microscopy", "stm", "atomic force microscopy", "afm",
        "transmission electron microscopy", "tem", "scanning electron microscopy", "sem",
        "electron energy loss spectroscopy", "eels", "x-ray diffraction", "xrd",
        "x-ray photoelectron spectroscopy", "xps", "ultraviolet photoelectron spectroscopy", "ups",
        "raman spectroscopy", "infrared spectroscopy", "ftir", "nuclear magnetic resonance", "nmr",
        "hall measurement", "magnetotransport", "photoconductivity", "time-resolved spectroscopy",
        "pump-probe", "transient absorption", "four-point probe", "kelvin probe",
        
        # Quantum Phenomena
        "spin", "valley", "quantum spin hall", "quantum anomalous hall", "spin wave",
        "magnon", "phonon", "phonon dispersion", "acoustic phonon", "optical phonon",
        "magneto-optical", "electro-optical", "kerr effect", "faraday effect", "rashba effect",
        "dresselhaus effect", "quantum size effect", "quantum capacitance", "quantum resistance",
        "landau level", "shubnikov-de haas", "de haas-van alphen", "aharonov-bohm",
        
        # Transport and Dynamic Properties
        "carrier lifetime", "recombination", "carrier mobility", "ballistic transport",
        "diffusive transport", "scattering", "mean free path", "drift", "diffusion",
        "auger recombination", "shockley-read-hall", "radiative recombination",
        "nonradiative recombination", "carrier injection", "thermionic emission",
        "tunneling", "fowler-nordheim tunneling", "direct tunneling", "trap-assisted tunneling",
        "hopping transport", "percolation", "anderson localization", "quantum capacitance",
        "interface state", "surface state", "trap state", "deep level", "donor level", "acceptor level",
        "defect state", "grain boundary", "dislocation", "vacancy", "interstitial",
        
        # New Device Concepts
        "neuromorphic", "memristor", "spintronic", "magnetic tunnel junction", "mtj",
        "valleytronics", "twistronic", "moiré pattern", "magic angle", "flat band",
        "correlated insulator", "unconventional superconductivity", "josephson junction",
        "squid", "single electron transistor", "resonant tunneling diode", "quantum cascade laser",
        "vertical cavity surface emitting laser", "vcsel", "distributed feedback laser", "dfb",
        "high electron mobility transistor", "hemt", "light emitting diode", "organic light emitting diode",
        "quantum light emitting diode", "field effect transistor", "thin film transistor",
        "lateral heterostructure", "vertical heterostructure", "van der waals heterostructure"
    }
    
    # Create case-insensitive version (convert all to lowercase)
    semiconductor_dictionary = {term.lower() for term in semiconductor_dictionary}
    
    # Initialize containers for tracking terms by source
    dictionary_terms = []  # Terms found via dictionary
    ner_terms = []         # Terms found via NER
    
    total_words = len(text.split())
    
    # 1. Dictionary-based identification
    # Tokenize text for basic word-level matching
    words = re.findall(r'\b[a-zA-Z0-9][\w\-\.]*[a-zA-Z0-9]\b|\b[a-zA-Z0-9]\b', text.lower())
    
    # Check single words against dictionary
    for word in words:
        if word.lower() in semiconductor_dictionary:
            dictionary_terms.append(word.lower())
    
    # Check for multi-word terms from the dictionary
    for term in semiconductor_dictionary:
        if ' ' in term and term.lower() in text.lower():
            # Count each occurrence
            count = text.lower().count(term.lower())
            for _ in range(count):
                dictionary_terms.append(term.lower())
    
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
        
        # 2.3. Pattern matching for chemical formulas, measurements, etc.
        chemical_formula_pattern = r'\b[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)+\b'
        measurement_pattern = r'\b\d+(?:\.\d+)?(?:n|µ|m|k|M|G)?(?:m|A|V|W|Hz|eV|Ω|F|H)\b'
        
        chemical_formulas = re.findall(chemical_formula_pattern, text)
        for formula in chemical_formulas:
            if formula.lower() not in semiconductor_dictionary:
                ner_terms.append(formula.lower())
                
        measurements = re.findall(measurement_pattern, text)
        for measurement in measurements:
            # Don't add simple numbers as technical terms
            if any(unit in measurement for unit in ['m', 'A', 'V', 'W', 'Hz', 'eV', 'Ω', 'F', 'H']):
                ner_terms.append(measurement.lower())
        
        # 3. Get syntax complexity score
        syntax_complexity_score = analyze_sentence_complexity_normalized(text, nlp)
        
    except Exception as e:
        print(f"Error in NLP-based technical term analysis: {str(e)}")
        syntax_complexity_score = 0.4  # Default value
    
    # Calculate metrics for each approach
    dictionary_count = len(dictionary_terms)
    ner_count = len(ner_terms)
    
    # Get LLM technical evaluation
    llm_evaluation = evaluate_technical_depth_with_llm(text)
    
    # Calculate total technical terms and CDI (Coverage Density Index)
    total_technical_terms = dictionary_count + ner_count
    cdi = total_technical_terms / (max(1, total_words) ** 0.5)  # Divide by square root of total words
    
    # Normalize CDI to 0-1 scale for weighted calculations
    # Using less stringent thresholds to allow for higher scores
    if cdi >= 5.0:
        normalized_cdi = 1.0      # Extremely high technical density
    elif cdi >= 4.0:
        normalized_cdi = 0.95     # Very high technical density
    elif cdi >= 3.0:
        normalized_cdi = 0.9      # High technical density
    elif cdi >= 2.5:
        normalized_cdi = 0.85     # Medium-high technical density
    elif cdi >= 2.0:
        normalized_cdi = 0.8      # Medium technical density
    elif cdi >= 1.5:
        normalized_cdi = 0.7      # Medium-low technical density
    elif cdi >= 1.0:
        normalized_cdi = 0.6      # Low technical density
    elif cdi >= 0.5:
        normalized_cdi = 0.4      # Very low technical density
    else:
        normalized_cdi = 0.2      # Minimal technical density
    
    # Adjust weights to reduce CDI weight and increase syntax and LLM weights
    cdi_weight = 1/3  # Equal weight (1/3)
    syntax_weight = 1/3  # Equal weight (1/3)
    llm_weight = 1/3  # Equal weight (1/3)
    
    # Calculate weighted score with equal weights
    balanced_technical_score = (
        (cdi_weight * normalized_cdi) + 
        (syntax_weight * syntax_complexity_score) +
        (llm_weight * llm_evaluation['score'])
    ) * 100  # Convert to percentage
    
    return {
        'dictionary_count': dictionary_count,
        'ner_count': ner_count,
        'total_words': total_words,
        'total_technical_terms': total_technical_terms,
        'cdi': cdi,
        'normalized_cdi': normalized_cdi,
        'normalized_dictionary_count': dictionary_count / max(1, total_words),
        'normalized_ner_count': ner_count / max(1, total_words),
        'syntax_complexity': syntax_complexity_score,
        'llm_evaluation': llm_evaluation,
        'balanced_technical_score': balanced_technical_score,
        'component_weights': {
            'cdi_weight': cdi_weight,
            'syntax_weight': syntax_weight,
            'llm_weight': llm_weight
        }
    }

# Add the new ContextualCoherenceAnalyzer class from final_evaluation_copy.py
class ContextualCoherenceAnalyzer:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
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

# Normalize Gunning Fog Index for technical content (target 12-14)
def normalize_gunning_fog(gunning_fog):
    """Normalize Gunning Fog Index to a 0-1 scale where 1.0 is optimal for technical writing"""
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

def evaluate_clarity_with_llm(text):
    """
    Use LLM to evaluate the clarity and understandability of the text.
    
    Returns:
        dict: Containing the clarity score and justification
    """
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
            return {
                'score': float(llm_evaluation.get('score', 0.5)),
                'justification': llm_evaluation.get('justification', "No justification provided")
            }
        else:
            return {
                'score': 0.5,
                'justification': "Failed to parse LLM response"
            }
            
    except Exception as e:
        print(f"Warning: LLM evaluation of clarity failed: {str(e)}")
        return {
            'score': 0.5,
            'justification': f"LLM evaluation failed: {str(e)}"
        }

def calculate_clarity(text):
    """
    Calculate clarity metrics combining Gunning Fog Index, coherence, and LLM evaluation
    with equal weights (1/3 each).
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary containing clarity metrics and scores
    """
    # 1. Calculate Gunning Fog Index
    gunning_fog = textstat.gunning_fog(text)
    normalized_fog = normalize_gunning_fog(gunning_fog)
    
    # 2. Get coherence metrics
    coherence_analyzer = ContextualCoherenceAnalyzer()
    coherence = coherence_analyzer.analyze_contextual_coherence(text)
    flow_score = coherence.get('concept_flow', {}).get('flow_score', 0.5)
    
    # Ensure flow_score is a valid number
    if np.isnan(flow_score):
        flow_score = 0.5  # Default to neutral score if NaN
    
    # 3. Get LLM clarity evaluation
    llm_evaluation = evaluate_clarity_with_llm(text)
    
    # 4. Calculate combined score with equal weights (1/3 each)
    combined_score = (normalized_fog + flow_score + llm_evaluation['score']) / 3.0
    
    # Return comprehensive results
    return {
        'gunning_fog': {
            'raw_score': gunning_fog,
            'normalized_score': normalized_fog
        },
        'coherence': coherence,
        'llm_evaluation': llm_evaluation,
        'combined_score': combined_score,
        'component_weights': {
            'gunning_fog': 1/3,
            'coherence': 1/3,
            'llm_evaluation': 1/3
        }
    }

def calculate_structure(text):
    """
    Calculate structure metrics for a given text, using topic hierarchy analysis
    and LLM evaluation with equal weights.
    
    Returns:
        dict: Dictionary containing structure metrics:
            - coherence: Contains concept_flow with score derived from topic hierarchy analysis
            - llm_evaluation: LLM-based structure evaluation
            - combined_score: Equally weighted combination of topic hierarchy and LLM score
            - component_weights: The weights used for each component
    """
    # Get topic hierarchy score (0-1)
    topic_hierarchy_score = analyze_topic_hierarchy_normalized(text)
    
    # Ensure topic_hierarchy_score is a valid number
    if np.isnan(topic_hierarchy_score):
        topic_hierarchy_score = 0.5  # Default to neutral score if NaN
    
    # Get LLM evaluation for structure
    llm_evaluation = evaluate_structure_with_llm(text)
    
    # Calculate combined score with equal weights (50% each)
    combined_score = (topic_hierarchy_score + llm_evaluation['score']) / 2.0
    
    # Create a coherence object that mimics the format of the original ContextualCoherenceAnalyzer
    # but uses the topic hierarchy score internally
    coherence = {
        'concept_flow': {
            'flow_score': float(topic_hierarchy_score),  # Use topic hierarchy score as flow score
            'quality': get_quality_label(topic_hierarchy_score),
            'details': {
                'local_coherence': {
                    'score': float(topic_hierarchy_score),
                    'assessment': get_quality_label(topic_hierarchy_score)
                },
                'progression': {
                    'score': float(topic_hierarchy_score),
                    'assessment': get_quality_label(topic_hierarchy_score)
                }
            }
        }
    }
    
    # Return comprehensive results with the original structure
    return {
        'coherence': coherence,  # Keep original 'coherence' key for backward compatibility
        'llm_evaluation': llm_evaluation,
        'combined_score': combined_score,
        'component_weights': {
            'coherence': 0.5,  # Keep original key name for backward compatibility
            'llm_evaluation': 0.5
        },
        # Add new fields without breaking backward compatibility
        'topic_hierarchy_score': topic_hierarchy_score,
        'internal_implementation': 'topic_hierarchy'  # Document what's happening internally
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

def evaluate_structure_with_llm(text):
    """
    Use LLM to evaluate the structural organization of the text.
    
    Returns:
        dict: Containing the structure score and justification
    """
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
            return {
                'score': float(llm_evaluation.get('score', 0.5)),
                'justification': llm_evaluation.get('justification', "No justification provided")
            }
        else:
            return {
                'score': 0.5,
                'justification': "Failed to parse LLM response"
            }
            
    except Exception as e:
        print(f"Warning: LLM evaluation of structure failed: {str(e)}")
        return {
            'score': 0.5,
            'justification': f"LLM evaluation failed: {str(e)}"
        }

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

Please analyze the text's factual accuracy by DIRECTLY MATCHING in-text citations with their corresponding reference data:
1. For EACH in-text citation (e.g., [1], [2], etc.) in the text, find the matching reference with the same ID
2. Compare the claim made when citing that reference against the actual content from that specific reference 
3. Determine if the claim is supported by the corresponding reference data
4. Check for any citations that don't have a corresponding reference or vice versa

Format your response as a JSON object with this structure:
{{
    "score": <0.0-1.0>,  # Overall factual accuracy score
    "analysis": [
        {{
            "claim": "specific claim or statement from text",
            "citation_id": "[X]",  # The exact citation ID used in the text
            "accuracy": <0.0-1.0>,  # Accuracy score for this claim
            "reference_support": "relevant information from reference [X] or 'Not supported'",
            "explanation": "brief explanation of accuracy rating"
        }},
        # Additional claims...
    ],
    "needs_improvement": <true/false>,
    "improvement_suggestions": "specific suggestions for improving factual accuracy"
}}

IMPORTANT: 
- Focus ONLY on claims with explicit citations
- Match each citation ID in the text to the same ID in the reference data
- Keep all claims and explanations short and focused
- Assign a SINGLE score that reflects the OVERALL factual accuracy
- Analyze ALL claims with citations in the text

Scoring criteria:
- 1.0: All claims are fully supported by their corresponding references
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

# Helper function to get quality labels for scores (mimics the original analyzer)
def get_quality_label(score):
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



