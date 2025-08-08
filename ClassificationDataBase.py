from itertools import combinations
import numpy as np










def pairwise_agreement(run_predictions: dict[str, list]) -> float:

    """
    What it measures
    How often predictions from different runs of the same model agree with each other on the same input, on average.
    
    How it's calculated
    For each sample:
    
    Compare all pairs of runs (e.g., run₁ vs run₂, run₁ vs run₃, etc.).
    
    Count how many of those pairs predicted the same class.
    
    Divide the number of agreements by the total number of pairs.
    
    Average this value across all samples.


    Interpretation
    1.0 = All runs always make the same prediction → perfectly robust.
    
    0.0 = All runs disagree as much as possible → highly unstable model.
    
    """




    
    runs = list(run_predictions.values())
    n_samples = len(runs[0])
    n_runs = len(runs)

    agreements = []

    for i in range(n_samples):
        preds = [run[i] for run in runs]
        total_pairs = 0
        agree_count = 0
        for a, b in combinations(preds, 2):
            total_pairs += 1
            if a == b:
                agree_count += 1
        agreements.append(agree_count / total_pairs if total_pairs else 1.0)

    return float(np.mean(agreements))


from collections import Counter
from scipy.stats import entropy

def consensus_entropy(run_predictions: dict[str, list]) -> float:
    """
    ➤ What it measures
    The uncertainty or diversity in predictions across runs for each input. This is calculated using Shannon entropy over the class distribution.
    
    ➤ How it's calculated
    For each sample:
    
    Count how many runs predicted each class (build a distribution).
    
    Convert counts into probabilities.
    
    Compute entropy of that distribution.
    
    Average entropy across all samples.
    
    ➤ Interpretation
    Low entropy (≈ 0.0): All runs agree → high robustness.
    
    High entropy (up to log₂(C)): Predictions spread across different classes → model is inconsistent.
    
    For binary classification, max entropy is 1.0. For 3 classes, max is ~1.58 (log₂(3)).
    
    """


    
    runs = list(run_predictions.values())
    n_samples = len(runs[0])
    n_runs = len(runs)

    entropies = []

    for i in range(n_samples):
        preds = [run[i] for run in runs]
        label_counts = Counter(preds)
        probs = np.array(list(label_counts.values())) / n_runs
        entropies.append(entropy(probs, base=2))

    return float(np.mean(entropies))



def majority_vote_agreement(run_predictions: dict[str, list]) -> float:

    """
    ➤ What it measures
    How often all runs agree with the majority prediction for a given input.
    
    ➤ How it's calculated
    For each sample:
    
    Find the most common predicted class (majority vote).
    
    If all runs predicted that same label, count it as 1, else 0.
    
    Average over all samples.
    
    ➤ Interpretation
    1.0: All runs always agree with each other → very consistent.
    
    <1.0: Some inputs have disagreement among runs.
    
    Lower values imply frequent disagreement, even on the most commonly predicted class.
    """





    
    runs = list(run_predictions.values())
    n_samples = len(runs[0])

    agreements = []

    for i in range(n_samples):
        preds = [run[i] for run in runs]
        most_common_label, count = Counter(preds).most_common(1)[0]
        agreements.append(1.0 if count == len(preds) else 0.0)

    return float(np.mean(agreements))




def compute_robustness_metrics_per_model(run_predictions: dict[str, list]) -> dict[str, float]:
    return {
        "pairwise_agreement": pairwise_agreement(run_predictions),
        "consensus_entropy": consensus_entropy(run_predictions),
        "majority_vote_agreement": majority_vote_agreement(run_predictions),
    }


# Filter out runs for a model from the full data
def get_model_runs(data: dict[str, list], model_id: str) -> dict[str, list]:
    return {
        k: v for k, v in data.items() if k.startswith(f"{model_id}_run_")
    }

# Sample data
data = {
    "input": ["a", "b", "c"],
    "label": [1, 0, 1],
    "modelX_run_0": [1, 0, 1],
    "modelX_run_1": [1, 1, 1],
    "modelX_run_2": [1, 0, 0],
}

model_runs = get_model_runs(data, "modelX")
metrics = compute_robustness_metrics_per_model(model_runs)

from pprint import pprint
pprint(metrics)




