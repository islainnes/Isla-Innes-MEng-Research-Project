import re
import spacy
import numpy as np
import openai
import json
import textstat  # Add this import for Gunning Fog
from typing import Dict

# Initialize OpenAI client with the actual API key from final_evaluation_copy.py
openai_client = openai.OpenAI(api_key="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")

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
        combined_score = (
            0.7 * normalized_depth +
            0.3 * normalized_clauses
        )
        
        return combined_score
        
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
        # Fixed prompt that includes the text to evaluate
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

def count_technical_terms_with_ner(text):
    """
    Count and identify technical terms using a comprehensive semiconductor dictionary
    first, then supplement with NLP techniques. Returns a balanced technical score
    based on dictionary terms, NER terms, syntax complexity, and LLM evaluation.
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
    # Typical CDI ranges: <0.5 (very low), 0.5-1.0 (low), 1.0-2.0 (medium), 2.0-3.0 (high), >3.0 (very high)
    if cdi >= 3.0:
        normalized_cdi = 1.0
    elif cdi >= 2.0:
        normalized_cdi = 0.9
    elif cdi >= 1.5:
        normalized_cdi = 0.8
    elif cdi >= 1.0:
        normalized_cdi = 0.7
    elif cdi >= 0.5:
        normalized_cdi = 0.5
    else:
        normalized_cdi = 0.3
    
    # New component weights with CDI replacing dictionary and NER components
    cdi_weight = 0.60  # Combined weight of previous dictionary (0.35) and NER (0.25)
    syntax_weight = 0.15  # Keep the same
    llm_weight = 0.25  # Keep the same
    
    # Calculate weighted score with the new components
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

def calculate_structure(text):
    """
    Calculate structure metrics for a given text, combining coherence analysis
    and LLM evaluation with equal weights.
    
    Returns:
        dict: Dictionary containing structure metrics:
            - coherence: Contextual coherence metrics from ContextualCoherenceAnalyzer
            - llm_evaluation: LLM-based structure evaluation
            - combined_score: Equally weighted combination of coherence and LLM score
            - component_weights: The weights used for each component
    """
    # Get coherence metrics
    coherence_analyzer = ContextualCoherenceAnalyzer()
    coherence = coherence_analyzer.analyze_contextual_coherence(text)
    flow_score = coherence.get('concept_flow', {}).get('flow_score', 0.5)
    
    # Ensure flow_score is a valid number
    if np.isnan(flow_score):
        flow_score = 0.5  # Default to neutral score if NaN
    
    # Get LLM evaluation for structure
    llm_evaluation = evaluate_structure_with_llm(text)
    
    # Calculate combined score with equal weights (50% each)
    combined_score = (flow_score + llm_evaluation['score']) / 2.0
    
    # Return comprehensive results
    return {
        'coherence': coherence,
        'llm_evaluation': llm_evaluation,
        'combined_score': combined_score,
        'component_weights': {
            'coherence': 0.5,
            'llm_evaluation': 0.5
        }
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

    
 

