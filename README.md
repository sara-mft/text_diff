# LLM-Bench: 

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Build Status]()
![Code Coverage]()
![License]()
![Last Commit]()

**LLM-Bench** provides a modular and extensible framework for evaluating and comparing the performance of various Large Language Models (LLMs) across a suite of standardized tasks. The primary goal is to offer a unified interface for running reproducible benchmarks and a standard results aggregation.

The framework is built on a set of core abstractions, allowing to integrate new models, datasets, and evaluation metrics with minimal boilerplate.

---

## 📋 Table of Contents

* [Key Features](#-key-features)
* [Getting Started](#-getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Running a Benchmark](#-running-a-benchmark)
* [Adding a New Task or Feature](#-adding-a-new-task-or-feature)
    * [Workflow 1: Using an Existing Task](#workflow-1-using-an-existing-task)
    * [Workflow 2: Implementing a New Task](#workflow-2-implementing-a-new-task)
* [Project Structure](#-project-structure)


---

## ✨ Key Features

* **TODO**: Easily extend the framework with new models, datasets, and tasks by inheriting from base classes.


---

## 🔧 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites



### Installation

1.  **Clone the repository:**


2.  **Create a virtual environment (recommended):**


3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 Running a Benchmark on a specific task

All benchmarks are executed via the `run_benchmark_single_task.py` script, which takes a configuration file as its main argument.

1.  **Define your benchmark** in a `.yaml` file inside the `benchmarking_scripts/` directory. See the example below.

    <details>
    <summary><b>Example: <code>benchmarking_scripts/my_classification_config.yaml</code></b></summary>

    ```yaml
    task: "classification"
    dataset:
      name: "my_custom_sentiment_dataset"
      path: "datasets/my_sentiment_data.csv"
      split:
        train: 0.8
        test: 0.2
    models:
      - "openai/gpt-4"
      - "google/gemini-1.5-pro"
      - "anthropic/claude-3-opus"
    evaluation:
      metrics: ["accuracy", "f1_score"]
      output_dir: "results/sentiment_analysis_run_1"
    ```
    </details>

2.  **Execute the script** from the root directory of the project:

    ```bash
    python run_benchmark_single_task.py benchmarking_scripts/my_classification_config.yaml
    ```

    💡 **Pro Tip**: To add new models to a completed benchmark run without overwriting previous results, use the `--add_models` flag. This is useful for incremental evaluations.

    ```bash
    # This will only run the benchmark for 'meta-llama/Llama-3-70b' and merge the results
    # with the existing ones from the previous run.
    python run_benchmark_single_task.py --add_models benchmarking_scripts/my_updated_config.yaml
    ```

---

## 🧩 Adding a New Task or Feature

The framework is designed for easy extension. Follow the appropriate workflow below.

### Workflow 1: Using an Existing Task

If your new feature involves a custom dataset for a supported task (e.g., **Classification**, **Translation**), follow these steps:

-   [ ] **1. Prepare Your Dataset**: Ensure your data is cleaned and structured.
-   [ ] **2. Verify Dataset Format**: Check `src/datasets/` to see the expected data format and structure for the existing task you intend to use.
-   [ ] **3. Place Dataset**: Move your final dataset file(s) into the `datasets/` directory.
-   [ ] **4. Create Config File**: Define a new `.yaml` configuration file in `benchmarking_scripts/` that points to your new dataset and specifies the models to be evaluated.
-   [ ] **5. Launch Benchmark**: Run the script as described in the [Running a Benchmark](#-running-a-benchmark) section.

### Workflow 2: Implementing a New Task

If you need to introduce a completely new task (e.g., "Summarization", "Code Generation"), you will need to implement the core logic.

-   [ ] **1. Implement Dataset Class**: Create a new dataset handler in `src/datasets/`. Your class should inherit from `BaseDataset` and implement the required data loading and preprocessing methods.
-   [ ] **2. Implement Task Class**: In `src/tasks/`, create a new task class that inherits from `BaseTask`. This class orchestrates the logic of how a model should perform the task (e.g., formatting prompts, parsing outputs).
-   [ ] **3. Implement Benchmark Class**: In `src/benchmarking/`, create a new benchmark class inheriting from `BaseBenchmark`. This class ties the dataset and task together and manages the evaluation loop.
-   [ ] **4. (Optional) Add Evaluation Metrics**: If your task requires custom metrics, add new functions or classes in `src/metrics/`.
-   [ ] **5. Place Dataset & Create Config**: Follow steps 3 and 4 from Workflow 1.
-   [ ] **6. Launch Benchmark**: Run the script. The framework will dynamically load your new task based on the `task` key in your YAML configuration.

---

## 📂 Project Structure

A brief overview of the key directories in this project.
```bash
llm-bench/
├── benchmarking_scripts/   # YAML configuration files for benchmarks
├── datasets/               # Raw and processed datasets
├── results/                # Output directory for benchmark results
├── src/                    # Source code
│   ├── benchmarking/       # Core benchmark orchestration logic
│   ├── datasets/           # Dataset loading and processing classes
│   ├── metrics/            # Evaluation metric implementations
│   ├── models/             # Model API abstractions
│   └── tasks/              # Task-specific logic (e.g., classification, translation)
├── tests/                  # Unit and integration tests
├── requirements.txt        # Project dependencies
└── run_benchmark_single_task.py # Main execution script
```
