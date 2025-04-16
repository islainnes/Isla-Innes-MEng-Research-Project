from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import numpy as np
from gensim.models import KeyedVectors
import json
import matplotlib.pyplot as plt
import os
import anthropic

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
word2vec_model = api.load('word2vec-google-news-300')  # This might take a while to download first time
client = anthropic.Anthropic(
    api_key="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"
)

def calculate_metrics(candidate_text, reference_text):
    # Cosine similarity using sentence embeddings
    candidate_embedding = embedding_model.encode(candidate_text, convert_to_tensor=True)
    reference_embedding = embedding_model.encode(reference_text, convert_to_tensor=True)
    cosine_sim = cosine_similarity([candidate_embedding.cpu().numpy()], 
                                 [reference_embedding.cpu().numpy()])[0][0]
    
    # Calculate Word Mover's Distance
    # Preprocess texts into lowercase tokens
    candidate_tokens = [word.lower() for word in candidate_text.split()]
    reference_tokens = [word.lower() for word in reference_text.split()]
    
    try:
        wmd = word2vec_model.wmdistance(candidate_tokens, reference_tokens)
        # Convert WMD to similarity score (1 / (1 + WMD)) so higher is better
        wmd_similarity = 1 / (1 + wmd)
    except KeyError:
        # Handle cases where some words are not in the vocabulary
        wmd_similarity = 0
    
    return {
        'cosine_similarity': cosine_sim,
        'wmd_similarity': wmd_similarity
    }

def get_llm_vote(original_text, improved_text, golden_standard):
    """
    Get LLM's evaluation of which response is better
    """
    prompt = f"""Compare these two responses about quantum tunneling effects and determine which one is better. 
    Consider technical accuracy, clarity, and completeness compared to the reference text.

    Reference text:
    {golden_standard}

    Original response:
    {original_text}

    Improved response:
    {improved_text}

    Format your response as a JSON object with this exact structure:
    {{
        "winner": "original or improved",
        "original_score": <0-100 score for original>,
        "improved_score": <0-100 score for improved>,
        "justification": "detailed explanation of the decision",
        "key_differences": ["difference1", "difference2", ...]
    }}
    """

    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        evaluation = json.loads(response.content[0].text)
        return evaluation
    except Exception as e:
        print(f"Error in LLM evaluation: {str(e)}")
        return {
            'winner': 'error',
            'original_score': 0,
            'improved_score': 0,
            'justification': "Error in evaluation process",
            'key_differences': ["Unable to evaluate due to error"]
        }

def evaluate_responses(model_responses_file, gs_answers_file):
    # Load model responses
    with open(model_responses_file, 'r') as f:
        responses = json.load(f)
    
    # Load golden standard answers
    with open(gs_answers_file, 'r') as f:
        gs_data = json.load(f)
    
    results = []
    
    # Evaluate only the first 6 responses
    for response_data in responses[:6]:
        question_num = response_data["question_number"]
        gs_answer = gs_data["qa_pairs"][question_num - 1]["answer"]
        
        # Clean responses
        original_response = response_data["original_model_response"].replace("[INST]", "").replace("[/INST]", "")
        improved_response = response_data["improved_only_model_response"].replace("[INST]", "").replace("[/INST]", "")
        
        # Calculate embedding metrics
        original_metrics = calculate_metrics(original_response, gs_answer)
        improved_metrics = calculate_metrics(improved_response, gs_answer)
        
        # Get LLM evaluation
        llm_eval = get_llm_vote(original_response, improved_response, gs_answer)
        
        results.append({
            "question_number": question_num,
            "question": response_data["question"],
            "original_metrics": {
                k: float(v) for k, v in original_metrics.items()
            },
            "improved_metrics": {
                k: float(v) for k, v in improved_metrics.items()
            },
            "llm_evaluation": llm_eval,
            "difference": {
                "cosine_similarity": float(improved_metrics["cosine_similarity"] - original_metrics["cosine_similarity"])
            }
        })
        
        # Print progress
        print(f"Evaluated question {question_num}/6")
    
    return results

def create_summary_plots(results):
    # Calculate averages
    avg_original_cosine = np.mean([r["original_metrics"]["cosine_similarity"] for r in results])
    avg_improved_cosine = np.mean([r["improved_metrics"]["cosine_similarity"] for r in results])
    avg_original_wmd = np.mean([r["original_metrics"]["wmd_similarity"] for r in results])
    avg_improved_wmd = np.mean([r["improved_metrics"]["wmd_similarity"] for r in results])
    avg_original_llm = np.mean([r["llm_evaluation"]["original_score"] for r in results])
    avg_improved_llm = np.mean([r["llm_evaluation"]["improved_score"] for r in results])
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Set up bar positions
    metrics = ['Cosine Similarity', 'WMD Similarity', 'LLM Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars
    original_bars = ax.bar(x - width/2, 
                          [avg_original_cosine, avg_original_wmd, avg_original_llm/100], 
                          width, 
                          label='Original Model',
                          color='#1f77b4',
                          alpha=0.8)
    
    improved_bars = ax.bar(x + width/2, 
                          [avg_improved_cosine, avg_improved_wmd, avg_improved_llm/100], 
                          width, 
                          label='Improved Model',
                          color='#2ca02c',
                          alpha=0.8)
    
    # Customize plot
    ax.set_ylabel('Average Score', fontsize=18)
    ax.set_title('Average Model Performance Comparison', fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', 
                       fontsize=14, fontweight='bold')
    
    autolabel(original_bars)
    autolabel(improved_bars)
    
    # Set y-axis limit with some padding
    plt.ylim(0, max([avg_original_cosine, avg_improved_cosine, 
                    avg_original_wmd, avg_improved_wmd, 
                    avg_original_llm/100, avg_improved_llm/100]) * 1.15)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Find the most recent model responses file
    model_comparisons_dir = "model_comparisons"
    response_files = [f for f in os.listdir(model_comparisons_dir) if f.startswith("model_responses_")]
    latest_file = max(response_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Evaluate responses
    results = evaluate_responses(
        os.path.join(model_comparisons_dir, latest_file),
        "gs_answers.json"
    )
    
    # Save results
    output_file = os.path.join(model_comparisons_dir, "evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create and save visualization
    create_summary_plots(results)
    
    # Print summary statistics
    avg_original_cosine = np.mean([r["original_metrics"]["cosine_similarity"] for r in results])
    avg_improved_cosine = np.mean([r["improved_metrics"]["cosine_similarity"] for r in results])
    avg_original_wmd = np.mean([r["original_metrics"]["wmd_similarity"] for r in results])
    avg_improved_wmd = np.mean([r["improved_metrics"]["wmd_similarity"] for r in results])
    avg_original_llm = np.mean([r["llm_evaluation"]["original_score"] for r in results])
    avg_improved_llm = np.mean([r["llm_evaluation"]["improved_score"] for r in results])
    
    print("\n=== Evaluation Results (First 6 Questions) ===")
    print("Embedding Metrics:")
    print(f"Average Original Model Cosine Similarity: {avg_original_cosine:.4f}")
    print(f"Average Improved Model Cosine Similarity: {avg_improved_cosine:.4f}")
    print(f"Cosine Similarity Improvement: {(avg_improved_cosine - avg_original_cosine):.4f}")
    
    print("\nWord Mover Distance Metrics:")
    print(f"Average Original Model WMD Similarity: {avg_original_wmd:.4f}")
    print(f"Average Improved Model WMD Similarity: {avg_improved_wmd:.4f}")
    print(f"WMD Similarity Improvement: {(avg_improved_wmd - avg_original_wmd):.4f}")
    
    print("\nLLM Evaluation:")
    print(f"Average Original Model Score: {avg_original_llm:.4f}")
    print(f"Average Improved Model Score: {avg_improved_llm:.4f}")
    print(f"Score Improvement: {(avg_improved_llm - avg_original_llm):.4f}")
    
    # Count LLM winners
    winners = [r["llm_evaluation"]["winner"] for r in results]
    print(f"\nLLM Winner Distribution:")
    print(f"Original: {winners.count('original')}")
    print(f"Improved: {winners.count('improved')}")
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"Visualization saved as: model_comparison_results.png")

if __name__ == "__main__":
    main()
