
from collections import Counter
from typing import List, Dict

class RobustnessMetric:
    """
    Metric to evaluate prediction robustness of an LLM-based classifier.
    Robustness is defined as consistency across repeated predictions.
    """

    def __init__(self, n_runs: int = 5):
        self.n_runs = n_runs

    def evaluate(self, predictions: List[str]) -> float:
        """
        Evaluate the robustness for a single prompt's predictions.

        Args:
            predictions (List[str]): List of predictions from n runs of the same prompt.

        Returns:
            float: Robustness score between 0 and 1.
        """
        if len(predictions) != self.n_runs:
            raise ValueError(f"Expected {self.n_runs} predictions, got {len(predictions)}")

        # Count the frequency of each unique prediction
        counts = Counter(predictions)
        most_common_freq = counts.most_common(1)[0][1]

        # Robustness: proportion of times the most frequent output occurred
        robustness_score = (most_common_freq - 1) / (self.n_runs - 1)
        return round(robustness_score, 4)

    def batch_evaluate(self, batch_predictions: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate robustness across multiple prompts.

        Args:
            batch_predictions (List[List[str]]): Each inner list contains n predictions for a single prompt.

        Returns:
            Dict[str, float]: Dictionary containing the average robustness and individual prompt scores.
        """
        scores = [self.evaluate(preds) for preds in batch_predictions]
        avg_score = round(sum(scores) / len(scores), 4)
        return {
            "average_robustness": avg_score,
            "individual_scores": scores
        }


# Exemple of usage
#if __name__ == "__main__":
#    metric = RobustnessMetric(n_runs=5)



    #predictions = [
    #    ["cat", "cat", "cat", "cat", "cat"],       # perfect consistency
    #    ["dog", "dog", "cat", "dog", "cat"],       # 3/5 same
    #    ["fish", "bird", "fish", "bird", "cat"],   # inconsistent
    #]

    #result = metric.batch_evaluate(predictions)
    #print(result)

