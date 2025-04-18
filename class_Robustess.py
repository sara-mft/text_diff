import json
from typing import List, Callable, Dict
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from openai.types.chat import ChatCompletion



class LLMRobustnessEvaluatorLLM:
    def __init__(
        self,
        generation_model: Callable[[str, int], List[str]],
        eval_model: Callable[[str], str],
        judge_model: Callable[[str, str], float]  # returns float in [0, 1]
    ):
        self.generate_variations = generation_model
        self.eval_model = eval_model
        self.judge_model = judge_model

    def evaluate_prompt(self, prompt: str, num_variations: int = 3) -> Dict:
        variations = self.generate_variations(prompt, num_variations)
        original_response = self.eval_model(prompt)

        results = {
            "original_prompt": prompt,
            "original_output": original_response,
            "variations": []
        }

        for variation in variations:
            output = self.eval_model(variation)
            similarity = self.judge_model(original_response, output)

            results["variations"].append({
                "perturbed_prompt": variation,
                "output": output,
                "llm_judge_similarity": similarity
            })

        return results

    def evaluate_batch(self, prompts: List[str], output_path: str = None) -> List[Dict]:
        all_results = [self.evaluate_prompt(p) for p in prompts]

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Robustness results saved to {output_path}")

        return all_results
    
    def llm_judge_similarity(response_a: str, response_b: str, model_name="gpt-4") -> float:
        client = OpenAI()

        prompt = (
            "You are an impartial judge. Rate how similar the following two responses are in terms of meaning (not wording). "
            "Use a score from 1 (completely different) to 5 (identical meaning).\n\n"
            f"Response A:\n{response_a}\n\nResponse B:\n{response_b}\n\nScore (1-5):"
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=10
            )
            content = response.choices[0].message.content.strip()
            score = int("".join(filter(str.isdigit, content)))
            return min(max(score, 1), 5) / 5.0  # normalize to [0, 1]

        except Exception as e:
            print(f"[Judge Error] Could not compute similarity: {e}")
            return 0.0
        



def main():
    evaluator = LLMRobustnessEvaluatorLLM(
        generation_model=openai_variation_generator,
        eval_model=openai_gpt4_call,
        judge_model=llm_judge_similarity
    )

    results = evaluator.evaluate_batch(
        prompts=[
            "Quelle est la classe de ce texte : 'Je recommande le docteur'"
        ],
        output_path="outputs/robustness_llm_judge.json"
    )