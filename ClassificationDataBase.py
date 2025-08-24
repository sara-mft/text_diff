from typing import List, Dict
import numpy as np
from collections import Counter

try:
    from lepor import Lepor
except ImportError:
    raise ImportError("Please install the 'lepor' package: pip install lepor")


def ngrams(sequence: List[str], n: int):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]


def compute_bleu(references: List[str], hypotheses: List[str], max_n: int = 4) -> float:
    """Compute a simple corpus BLEU score."""
    weights = [1.0/max_n] * max_n
    clipped_counts, total_counts = [0]*max_n, [0]*max_n
    ref_length, hyp_length = 0, 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)

        for n in range(1, max_n+1):
            ref_ngrams = Counter(ngrams(ref_tokens, n))
            hyp_ngrams = Counter(ngrams(hyp_tokens, n))
            overlap = {ng: min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items()}
            clipped_counts[n-1] += sum(overlap.values())
            total_counts[n-1] += max(sum(hyp_ngrams.values()), 1)

    precisions = [clipped_counts[i]/total_counts[i] if total_counts[i] > 0 else 0 for i in range(max_n)]
    if min(precisions) == 0:
        return 0.0

    log_precisions = sum([w * np.log(p) for w, p in zip(weights, precisions)])
    bp = np.exp(1 - ref_length/hyp_length) if hyp_length < ref_length else 1.0
    return float(bp * np.exp(log_precisions)) * 100


def compute_nist(references: List[str], hypotheses: List[str], max_n: int = 5) -> float:
    """Compute a simplified NIST score (information-weighted BLEU)."""
    total_info, total_count = 0.0, 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        for n in range(1, max_n+1):
            ref_ngrams = Counter(ngrams(ref_tokens, n))
            hyp_ngrams = Counter(ngrams(hyp_tokens, n))
            for ng, count in hyp_ngrams.items():
                if ng in ref_ngrams:
                    info_weight = -np.log2((ref_ngrams[ng] + 1) / (len(ref_tokens) - n + 1))
                    total_info += count * info_weight
                    total_count += count

    return (total_info / max(total_count, 1)) * 100


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    dp = np.zeros((len(ref)+1, len(hyp)+1), dtype=int)
    for i in range(len(ref)+1):
        dp[i][0] = i
    for j in range(len(hyp)+1):
        dp[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len(ref)][len(hyp)]


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Word Error Rate (WER)."""
    total_err, total_words = 0, 0
    for ref, hyp in zip(references, hypotheses):
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        total_err += edit_distance(ref_tokens, hyp_tokens)
        total_words += len(ref_tokens)
    return (total_err / total_words) * 100


def compute_meteor(references: List[str], hypotheses: List[str]) -> float:
    """Simplified METEOR implementation (no synonym matching)."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        overlap = len(set(ref_tokens) & set(hyp_tokens))
        if overlap == 0:
            scores.append(0)
            continue
        precision = overlap / len(hyp_tokens)
        recall = overlap / len(ref_tokens)
        f_mean = (10 * precision * recall) / (recall + 9*precision) if (precision+recall)>0 else 0
        frag_penalty = 0.5 * (overlap / (overlap + abs(len(ref_tokens)-len(hyp_tokens))))
        scores.append(f_mean * frag_penalty)
    return float(np.mean(scores)) * 100


def compute_lepor(references: List[str], hypotheses: List[str]) -> float:
    lepor_scorer = Lepor()
    scores = [lepor_scorer.lepor(hyp.split(), ref.split()) for hyp, ref in zip(hypotheses, references)]
    return float(np.mean(scores)) * 100


def evaluate_mt(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    return {
        "BLEU": compute_bleu(references, hypotheses),
        "NIST": compute_nist(references, hypotheses),
        "WER": compute_wer(references, hypotheses),
        "METEOR": compute_meteor(references, hypotheses),
        "LEPOR": compute_lepor(references, hypotheses),
    }


if __name__ == "__main__":
    references = [
        "The cat is on the mat.",
        "There is a cat sitting on the mat.",
    ]
    hypotheses = [
        "The cat sits on the mat.",
        "A cat is on the mat.",
    ]

    results = evaluate_mt(references, hypotheses)
    print("\nMachine Translation Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
