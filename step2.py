import json
import numpy as np
from textstat import textstat
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import nltk
import os
from pathlib import Path

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

def calculate_readability_metrics(text):
    """Calculate readability metrics for the text"""
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'text_standard': textstat.text_standard(text),
        'syllable_stats': {
            'syllable_count': textstat.syllable_count(text),
            'polysyllable_count': textstat.polysyllabcount(text),
            'avg_syllables_per_word': textstat.avg_syllables_per_word(text)
        },
        'sentence_stats': {
            'sentence_count': textstat.sentence_count(text),
            'avg_sentence_length': textstat.avg_sentence_length(text),
            'avg_sentence_per_word': textstat.avg_sentence_per_word(text)
        },
        'lexical_stats': {
            'lexicon_count': textstat.lexicon_count(text),
            'difficult_words': textstat.difficult_words(text)
        }
    }

def calculate_technical_depth(text):
    """Calculate technical depth metrics"""
    technical_terms = [
        'semiconductor', 'transistor', 'electron', 'voltage', 'current',
        'thermal', 'quantum', 'spin', 'carrier', 'doping', 'oxide',
        'silicon', 'device', 'fabrication', 'manufacturing', 'yield',
        'power', 'module', 'gate', 'material'
    ]
    
    term_counts = {term: len(re.findall(rf'\b{term}\b', text, re.I)) 
                  for term in technical_terms}
    
    return {
        'technical_term_count': sum(term_counts.values()),
        'technical_terms_detail': term_counts,
        'citation_count': len(re.findall(r'\[\d+\]', text)),
        'technical_density': sum(term_counts.values()) / len(text.split())
    }

def calculate_structure_metrics(text):
    """Calculate structural metrics of the text"""
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    sentences = nltk.sent_tokenize(text)
    
    return {
        'paragraph_count': len(paragraphs),
        'avg_paragraph_length': np.mean([len(p.split()) for p in paragraphs]),
        'sentence_count': len(sentences),
        'avg_sentence_length': np.mean([len(s.split()) for s in sentences]),
        'word_count': len(text.split()),
        'has_topic_sentences': sum(1 for p in paragraphs if re.match(r'^[^.]+\b(?:discuss|present|analyze|examine|investigate|explore)\b', p.strip()))
    }

def calculate_critical_thinking_metrics(text):
    """Calculate critical thinking metrics"""
    critical_phrases = [
        'however', 'although', 'nevertheless', 'contrary to',
        'limitation', 'gap', 'weakness', 'strength',
        'inconsistent', 'contradictory', 'debate', 'controversy'
    ]
    
    critical_counts = {phrase: len(re.findall(rf'\b{phrase}\b', text, re.I)) 
                      for phrase in critical_phrases}
    
    synthesis_phrases = [
        'in conclusion', 'overall', 'taken together',
        'this suggests', 'these findings indicate',
        'across studies', 'the literature shows'
    ]
    
    return {
        'critical_metrics': {
            'critical_phrase_count': sum(critical_counts.values()),
            'critical_phrase_detail': critical_counts,
            'critical_density': sum(critical_counts.values()) / len(text.split())
        },
        'synthesis_metrics': {
            'synthesis_phrase_count': sum(1 for phrase in synthesis_phrases 
                                       if phrase in text.lower()),
            'integration_indicators': {
                'comparison_count': len(re.findall(r'\b(?:compare|contrast|while|whereas)\b', text, re.I)),
                'synthesis_count': len(re.findall(r'\b(?:combine|integrate|synthesize)\b', text, re.I))
            }
        }
    }

def calculate_similarity_to_papers(text, papers, encoder):
    """Calculate similarity between text and referenced papers"""
    text_embedding = encoder.encode(text)
    similarities = []
    
    for paper in papers:
        paper_title = paper['title']
        paper_year = paper['year']
        similarity = paper['similarity']
        
        similarities.append({
            'title': paper_title,
            'year': paper_year,
            'similarity_score': similarity
        })
    
    return {
        'paper_similarities': similarities,
        'avg_similarity': np.mean([s['similarity_score'] for s in similarities]),
        'max_similarity': max(similarities, key=lambda x: x['similarity_score']),
        'min_similarity': min(similarities, key=lambda x: x['similarity_score'])
    }

def analyze_report(report_path):
    """Analyze the report and calculate all metrics"""
    # Load the report
    with open(report_path, 'r') as f:
        report_data = json.load(f)
    
    # Initialize the sentence transformer model
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize sentiment analyzer if CUDA is available
    device = 0 if torch.cuda.is_available() else -1
    sentiment_analyzer = pipeline("sentiment-analysis",
                                model="nlptown/bert-base-multilingual-uncased-sentiment",
                                device=device)
    
    # Get sections and papers with default values if missing
    sections = report_data.get('sections', {})
    papers = report_data.get('referenced_papers', [])
    
    # Calculate metrics for each section
    metrics = {
        'metadata': {
            'topic': report_data.get('topic', 'Unknown'),
            'timestamp': report_data.get('timestamp', None)
        },
        'sections': {}
    }
    
    for section_name, content in sections.items():
        section_metrics = {
            'readability': calculate_readability_metrics(content),
            'technical_depth': calculate_technical_depth(content),
            'structure': calculate_structure_metrics(content),
            'critical_thinking': calculate_critical_thinking_metrics(content),
            'similarity': calculate_similarity_to_papers(content, papers, encoder)
        }
        
        # Calculate sentiment
        sentences = nltk.sent_tokenize(content)
        sentiments = sentiment_analyzer(sentences[:100])  # Limit to first 100 sentences
        sentiment_scores = {
            'positive': len([s for s in sentiments if s['label'] in ['5 stars', '4 stars']]),
            'neutral': len([s for s in sentiments if s['label'] == '3 stars']),
            'negative': len([s for s in sentiments if s['label'] in ['1 star', '2 stars']])
        }
        section_metrics['sentiment'] = sentiment_scores
        
        metrics['sections'][section_name] = section_metrics
    
    # Calculate overall metrics
    full_text = '\n\n'.join(sections.values())
    metrics['overall'] = {
        'readability': calculate_readability_metrics(full_text),
        'technical_depth': calculate_technical_depth(full_text),
        'structure': calculate_structure_metrics(full_text),
        'critical_thinking': calculate_critical_thinking_metrics(full_text),
        'similarity': calculate_similarity_to_papers(full_text, papers, encoder)
    }
    
    return metrics

def main():
    """Process all report files in the litreviews folder"""
    # Create output directory if it doesn't exist
    output_dir = Path('report_metrics')
    output_dir.mkdir(exist_ok=True)

    # Process all JSON files in the litreviews folder
    litreviews_dir = Path('litreviews')
    for report_file in litreviews_dir.glob('*.json'):
        try:
            # Analyze the report
            print(f"Processing {report_file.name}...")
            metrics = analyze_report(str(report_file))
            
            # Create output filename based on input filename
            output_path = output_dir / f"{report_file.stem}_metrics.json"
            
            # Save the metrics to a JSON file
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Metrics saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {report_file.name}: {str(e)}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
