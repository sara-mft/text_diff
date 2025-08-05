import os
import random
import pandas as pd
from typing import List, Dict, Union, Iterator, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationDataset:
    def __init__(self, data_dir: str, data_name: str, **kwargs):
        self.data_dir = data_dir
        self.data_name = data_name
        self.metadata_dict = {
            "name": kwargs.get("name"),
            "version": kwargs.get("version"),
            "description": kwargs.get("description"),
            "citation": kwargs.get("citation"),
            "language": kwargs.get("language", []),
            "link": kwargs.get("link"),
            "licence": kwargs.get("licence"),
            "internal_project_name": kwargs.get("internal_project_name"),
            "class_labels": kwargs.get("class_labels", [])
        }

        # Internal data containers
        self._dataframe: Optional[pd.DataFrame] = None
        self._data_records: Optional[List[Dict[str, str]]] = None

        self._validate_language()
        self._validate_labels()

    def _validate_language(self):
        allowed_languages = {"multilingual", "fr", "en"}
        if not set(self.metadata_dict["language"]).issubset(allowed_languages):
            raise ValueError(f"Unsupported languages found. Allowed: {allowed_languages}")

    def _validate_labels(self):
        labels = self.metadata_dict["class_labels"]
        if labels and not isinstance(labels, list):
            raise ValueError("class_labels must be a list of strings.")
    
    def metadata(self) -> Dict:
        """Return the metadata information as a dictionary."""
        return self.metadata_dict

    def load_data(self, data_path: Optional[str] = None) -> Union[pd.DataFrame, List[Dict[str, str]]]:
        """
        Load an Excel file with 'input' and 'label' columns.
        Returns both a DataFrame and a list of dicts.
        """
        full_path = data_path or os.path.join(self.data_dir, self.data_name)
        
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Data file not found at: {full_path}")
        
        try:
            df = pd.read_excel(full_path, engine='openpyxl')
        except Exception as e:
            logger.error("Failed to read Excel file.")
            raise e

        expected_columns = {"input", "label"}
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"Excel file must contain columns: {expected_columns}")

        df = df.dropna(subset=["input", "label"])
        df["input"] = df["input"].astype(str)
        df["label"] = df["label"].astype(str)

        self._dataframe = df.reset_index(drop=True)
        self._data_records = self._dataframe.to_dict(orient="records")

        logger.info(f"Loaded {len(self._dataframe)} rows from {full_path}")
        return self._dataframe, self._data_records

    def get_data_sample(self) -> Dict[str, str]:
        """Return a single random data sample."""
        if self._data_records is None:
            raise RuntimeError("Data not loaded. Call `load_data()` first.")
        
        sample = random.choice(self._data_records)
        return {"input": sample["input"], "label": sample["label"]}

    def get_data_iterator(self, shuffle: bool = True) -> Iterator[Dict[str, str]]:
        """Return an iterator over data samples."""
        if self._data_records is None:
            raise RuntimeError("Data not loaded. Call `load_data()` first.")

        records = self._data_records.copy()
        if shuffle:
            random.shuffle(records)

        for record in records:
            yield {"input": record["input"], "label": record["label"]}



    def data_stats(self) -> Dict:
        """Return dataset statistics as a dictionary."""
        if self._dataframe is None:
            raise RuntimeError("Data not loaded. Call `load_data()` first.")

        df = self._dataframe.copy()

        input_lengths_chars = df["input"].apply(len)
        input_lengths_words = df["input"].apply(lambda x: len(str(x).split()))
        label_lengths_chars = df["label"].apply(lambda x: len(str(x)))

        stats = {
            "num_rows": len(df),
            "num_classes": df["label"].nunique(),
            "class_distribution": df["label"].value_counts().to_dict(),
            "input_avg_length_chars": input_lengths_chars.mean(),
            "input_avg_length_words": input_lengths_words.mean(),
            "label_avg_length_chars": label_lengths_chars.mean(),
            "unique_inputs": df["input"].nunique(),
            "unique_labels": df["label"].nunique(),
            "input_max_length": input_lengths_chars.max(),
            "input_min_length": input_lengths_chars.min(),
            "duplicates": df.duplicated(subset=["input", "label"]).sum(),
            "missing_values": df.isnull().sum().to_dict(),
        }

        return stats



















################################


dataset = ClassificationDataset(
    data_dir="data/",
    data_name="my_dataset.xlsx",
    name="MyDataset",
    version="1.0",
    description="Example classification dataset.",
    language=["en"],
    class_labels=["positive", "negative"]
)

# Load data
df, data_dicts = dataset.load_data()

# Get metadata
print(dataset.metadata())

# Get random sample
print(dataset.get_data_sample())

# Iterate over data
for item in dataset.get_data_iterator():
    print(item)
    break  # just show one

# Get stats
stats = dataset.data_stats()
for key, value in stats.items():
    print(f"{key}: {value}")

