import os
import json
import time
import argparse
from typing import Callable, Dict, List
from datasets import load_dataset
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

# --- Core Benchmarking Class ---
class IFEvalBenchmark:
    def __init__(
        self,
        dataset_name: str = "le-leadboard/IFEval-fr",
        subset: str = "default",
        split: str = "test",
        cache_dir: str = "./.cache"
    ):
        self.dataset = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
        self.results: Dict[str, List[Dict]] = {}

    def run_multiple_models(
        self,
        models: Dict[str, Callable[[str], str]],
        num_samples: int = 10,
        sleep: float = 0.5
    ):
        for model_name, model_call in models.items():
            print(f"⚙️ Running model: {model_name}")
            responses = []

            for idx, sample in enumerate(self.dataset.select(range(num_samples))):
                instruction = sample["instruction"]
                reference = sample["output"]

                try:
                    prediction = model_call(instruction)
                except Exception as e:
                    print(f"[{model_name}] ❌ Error at index {idx}: {e}")
                    prediction = ""

                responses.append({
                    "id": sample["id"],
                    "instruction": instruction,
                    "reference": reference,
                    "prediction": prediction
                })

                if idx % 5 == 0:
                    print(f"[{model_name}] ✅ Processed {idx + 1}/{num_samples}")

                if sleep:
                    time.sleep(sleep)

            self.results[model_name] = responses


            

    def save_results(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"📁 Results saved to: {output_path}")

    def compare_side_by_side(self, model_names: List[str], sample_count: int = 5):
        print("\n📊 Side-by-side Comparison")
        for i in range(sample_count):
            print("=" * 80)
            print(f"[Sample #{i}]")
            instruction = self.results[model_names[0]][i]["instruction"]
            print(f"📝 Instruction:\n{instruction}\n")

            for model in model_names:
                print(f"🤖 {model} Prediction:\n{self.results[model][i]['prediction']}\n")

            print(f"✅ Reference:\n{self.results[model_names[0]][i]['reference']}")
            print("=" * 80 + "\n")


def openai_gpt4_call(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def azure_gpt4_call_factory(endpoint: str, api_key: str, deployment: str = "gpt-4o"):
    def azure_call(prompt: str) -> str:
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint
        )
        response: ChatCompletion = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    return azure_call


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs on IFEval-fr dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate.")
    parser.add_argument("--output", type=str, required=True, help="Path to save JSON output.")
    parser.add_argument("--use_openai", action="store_true", help="Include OpenAI GPT-4.")
    parser.add_argument("--use_azure", action="store_true", help="Include Azure GPT-4.")
    parser.add_argument("--azure_endpoint", type=str, default="", help="Azure endpoint URL.")
    parser.add_argument("--azure_key", type=str, default="", help="Azure API key.")
    parser.add_argument("--azure_deployment", type=str, default="gpt-4o", help="Azure deployment name.")
    parser.add_argument("--compare", action="store_true", help="Print side-by-side comparison.")

    args = parser.parse_args()

    benchmark = IFEvalBenchmark()

    model_calls = {}

    if args.use_openai:
        model_calls["gpt-4"] = openai_gpt4_call

    if args.use_azure:
        if not args.azure_endpoint or not args.azure_key:
            raise ValueError("Azure endpoint and API key must be provided.")
        model_calls["azure-gpt-4"] = azure_gpt4_call_factory(
            endpoint=args.azure_endpoint,
            api_key=args.azure_key,
            deployment=args.azure_deployment
        )

    if not model_calls:
        raise ValueError("At least one model must be specified.")

    benchmark.run_multiple_models(
        models=model_calls,
        num_samples=args.num_samples
    )

    benchmark.save_results(args.output)

    if args.compare:
        benchmark.compare_side_by_side(model_names=list(model_calls.keys()), sample_count=5)


if __name__ == "__main__":
    main()