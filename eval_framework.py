from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score
from textstat import textstat
import json
import nltk
nltk.download('punkt')  # Required for BLEU score

# Initialize scoring tools
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def get_readability_metrics(text):
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text)
    }

def structural_metrics(text):
    return {
        'word_count': len(text.split()),
        'sentence_count': len(nltk.sent_tokenize(text)),
        'avg_sentence_length': len(text.split()) / max(len(nltk.sent_tokenize(text)), 1)
    }

def calculate_metrics(candidate_text, reference_text):
    # Cosine similarity
    candidate_embedding = embedding_model.encode(candidate_text, convert_to_tensor=True)
    reference_embedding = embedding_model.encode(reference_text, convert_to_tensor=True)
    cosine_sim = cosine_similarity([candidate_embedding.cpu().numpy()], 
                                 [reference_embedding.cpu().numpy()])[0][0]
    
    # ROUGE scores
    rouge_scores = rouge.score(reference_text, candidate_text)
    
    # BLEU score
    reference_tokens = [reference_text.split()]
    candidate_tokens = candidate_text.split()
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    
    # BERTScore
    P, R, F1 = score([candidate_text], [reference_text], lang='en', verbose=False)
    bert_f1 = F1.mean().item()
    
    # Readability and structural metrics
    readability = get_readability_metrics(candidate_text)
    structure = structural_metrics(candidate_text)
    
    return {
        'cosine_similarity': cosine_sim,
        'rouge1_f1': rouge_scores['rouge1'].fmeasure,
        'rouge2_f1': rouge_scores['rouge2'].fmeasure,
        'rougeL_f1': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu_score,
        'bert_score': bert_f1,
        **readability,
        **structure
    }

# Load the golden standard
golden_standard = """Quantum tunneling effects significantly impact next-generation semiconductor devices in several key ways:

Bandgap Engineering and Quantum Confinement:
Quantum confinement dramatically affects the optoelectronic properties of semiconductor devices
The bandgap varies inversely with the size of quantum structures
This enables precise tuning of device characteristics through size control of quantum structures

Hot Carrier Effects:
Quantum tunneling influences carrier cooling rates and energy transfer mechanisms
In quantum structures like quantum wells and quantum dots, carrier relaxation dynamics are modified due to confinement effects
When carrier cooling rates become comparable to impact ionization rates, excess energy can be converted into additional photocurrent

Device Performance Implications:
Affects carrier lifetime control and switching characteristics in power semiconductor devices
Influences off-state leakage current and device reliability
Enables new functionalities through quantum mechanical effects that aren't possible in conventional devices

Applications in Advanced Devices:
Quantum Dot Solar Cells: Enhanced photovoltaic performance through controlled carrier dynamics
Quantum Cascade Lasers: Utilization of tunneling for controlled electron transport and emission
Spintronics: Enables spin-based logic and quantum information processing through controlled quantum tunneling

Efficiency Considerations:
Can help reduce thermalization losses in solar cells
Enables multiple exciton generation in quantum dot devices
Allows for better control of carrier transport and recombination processes

These quantum tunneling effects are particularly important for:

High-efficiency photovoltaic devices
Next-generation computing devices
Quantum information processing systems
High-frequency electronic components

The understanding and control of these quantum tunneling effects is crucial for the continued development of more efficient and capable semiconductor devices."""

# Load and parse the JSON file
with open('model_comparisons/model_responses_1739891955.json', 'r') as f:
    responses = json.load(f)

# Get the first question's responses
first_question = responses[0]
models = {
    'Original': first_question["original_model_response"],
    'LitReviews': first_question["litreviews_model_response"],
    'Files MMD': first_question["files_mmd_model_response"]
}

# Calculate and display metrics for each model
print("=== Comprehensive Evaluation Metrics ===\n")
for model_name, response in models.items():
    metrics = calculate_metrics(response, golden_standard)
    print(f"\n{model_name} Model Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Compare differences between models
print("\n=== Model Comparisons ===")
baseline_metrics = calculate_metrics(models['Original'], golden_standard)
for model_name in ['LitReviews', 'Files MMD']:
    model_metrics = calculate_metrics(models[model_name], golden_standard)
    print(f"\n{model_name} vs Original differences:")
    print("-" * 30)
    for metric in model_metrics.keys():
        diff = model_metrics[metric] - baseline_metrics[metric]
        print(f"{metric}: {diff:+.4f}")
