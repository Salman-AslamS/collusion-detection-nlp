"""
experiment_logger.py
--------------------
Tracks and persists experiment results across the 53-run search space
used in the dissertation (INM373, City University London).

Each experiment logs its configuration, precision/recall/F1, processing
speed, and a sample of scored document pairs to a JSON master log with
automatic backup.
"""

import os
import json
from datetime import datetime

import pandas as pd


class ExperimentLogger:
    """
    Persistent experiment logger for LSH / BERT similarity experiments.

    Stores results in a JSON master log with per-experiment detail files.
    Safe to re-instantiate across sessions — loads existing log if found.

    Args:
        log_dir (str): Directory for log files (local path or Google Drive path).

    Example:
        >>> logger = ExperimentLogger("logs/")
        >>> logger.log_experiment(config, results, sample_pairs, notes="run 1")
        >>> df = logger.get_experiment_summary()
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.master_log_file = os.path.join(log_dir, "master_log.json")

        if os.path.exists(self.master_log_file):
            with open(self.master_log_file, "r") as f:
                self.master_log = json.load(f)
            print(f"Loaded {len(self.master_log)} existing experiments from {self.master_log_file}")
        else:
            self.master_log = []
            print(f"New experiment log created at {self.master_log_file}")

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        experiment_config: dict,
        results: dict,
        sample_pairs: list,
        notes: str = "",
    ) -> str:
        """
        Log a completed experiment run.

        Args:
            experiment_config: Dict with keys 'name' and 'params'.
            results: Dict with precision, recall, f1_score, processing_time,
                     total_documents, true_positives, false_positives, false_negatives.
            sample_pairs: List of dicts with scored document pair examples.
            notes: Free-text notes (model observations, anomalies, etc.).

        Returns:
            experiment_id (str): Timestamped unique identifier.
        """
        experiment_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_config['name']}"
        )
        docs_per_second = (
            float(results["total_documents"]) / float(results["processing_time"])
            if results["processing_time"] > 0
            else 0.0
        )

        record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": experiment_config,
            "results": {
                "precision": float(results["precision"]),
                "recall": float(results["recall"]),
                "f1_score": float(results["f1_score"]),
                "processing_time": float(results["processing_time"]),
                "documents_per_second": docs_per_second,
                "total_documents": results["total_documents"],
                "true_positives": results["true_positives"],
                "false_positives": results["false_positives"],
                "false_negatives": results["false_negatives"],
            },
            "sample_pairs": sample_pairs,
            "notes": notes,
        }

        self.master_log.append(record)
        self._save()

        # Also write a per-experiment detail file
        detail_path = os.path.join(self.log_dir, f"{experiment_id}.json")
        with open(detail_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"Logged experiment: {experiment_id}")
        return experiment_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_experiment_summary(self, experiment_id: str = None) -> pd.DataFrame:
        """
        Return experiment results as a DataFrame.

        Args:
            experiment_id: If given, return only that experiment's results.
                           If None, return all experiments.

        Returns:
            pd.DataFrame with one row per experiment.
        """
        if experiment_id:
            for exp in self.master_log:
                if exp["experiment_id"] == experiment_id:
                    return pd.DataFrame([exp["results"]])
            return pd.DataFrame()

        rows = []
        for exp in self.master_log:
            row = exp["results"].copy()
            row["experiment_id"] = exp["experiment_id"]
            row["configuration"] = str(exp["configuration"])
            row["timestamp"] = exp["timestamp"]
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self):
        """Write master log with backup of the previous version."""
        if os.path.exists(self.master_log_file):
            os.replace(self.master_log_file, self.master_log_file + ".backup")
        with open(self.master_log_file, "w") as f:
            json.dump(self.master_log, f, indent=2)
