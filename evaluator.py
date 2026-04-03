"""
evaluator.py
------------
Evaluation harness for LSH and Fusion similarity detectors.

Provides two functions:

    evaluate_lsh(detector, df, ...)    → Run OptimizedLSH or UltimateLSH
                                         on a dataset and return metrics.

    evaluate_fusion(model, df, ...)    → Run UltimateFusionModel on a
                                         dataset and return metrics.

Both functions return a standardised results dict and optionally log
the run via ExperimentLogger.

Dataset format expected:
    pd.DataFrame with columns ['original', 'paraphrase']
    where each row is a known similar pair (label = 1).

Evaluation note:
    Ground truth in this dataset is that every (original, paraphrase) pair
    IS similar (label = 1). Precision is measured as the proportion of
    flagged pairs that are correct. Recall is the proportion of true pairs
    that are detected. The academic integrity use case prioritises precision
    — a false accusation is more harmful than a missed detection.
"""

import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiment_logger import ExperimentLogger


# =============================================================================
# LSH evaluation
# =============================================================================

def evaluate_lsh(
    detector,
    df: pd.DataFrame,
    threshold: float = None,
    sample_size: int = None,
    log_dir: str = None,
    experiment_name: str = "lsh_evaluation",
) -> dict:
    """
    Evaluate an LSH-based detector on a labelled dataset.

    Args:
        detector: An OptimizedLSHDetector or UltimateLSHDetector instance.
        df: DataFrame with 'original' and 'paraphrase' columns.
        threshold: Similarity threshold for binary classification.
                   Defaults to detector.threshold.
        sample_size: Evaluate on a random subsample if given.
        log_dir: If provided, log results via ExperimentLogger.
        experiment_name: Name tag for the logged experiment.

    Returns:
        dict with precision, recall, f1_score, docs_per_second,
        processing_time, total_documents, and component stats.
    """
    threshold = threshold or detector.threshold
    eval_df = df.sample(n=sample_size, random_state=42) if sample_size else df
    pairs = list(zip(eval_df["original"], eval_df["paraphrase"]))

    print(f"Evaluating {experiment_name} on {len(pairs)} pairs...")

    similarities = []
    times = []

    for text1, text2 in tqdm(pairs, desc="Processing"):
        t0 = time.time()
        result = detector.get_document_similarity(text1, text2)
        # Handle both dict and float return types
        score = result["final_similarity"] if isinstance(result, dict) else float(result)
        similarities.append(score)
        times.append(time.time() - t0)

    total_time = sum(times)
    predictions = [1 if s >= threshold else 0 for s in similarities]

    # All pairs are true positives by construction; undetected = false negatives
    true_positives = sum(predictions)
    false_negatives = len(predictions) - true_positives
    false_positives = 0  # Would require negative pairs to measure

    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "processing_time": total_time,
        "docs_per_second": len(pairs) / total_time if total_time > 0 else 0,
        "total_documents": len(pairs),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "avg_similarity": float(np.mean(similarities)),
        "avg_time_ms": float(np.mean(times) * 1000),
    }

    _print_results(experiment_name, results)

    if log_dir:
        logger = ExperimentLogger(log_dir)
        config = {
            "name": experiment_name,
            "params": {
                "threshold": threshold,
                "num_perm": getattr(detector, "num_perm", None),
                "ngram_range": getattr(detector, "ngram_range", None),
                "weights": getattr(detector, "weights", None),
            },
        }
        sample_pairs = _build_sample_pairs(detector, pairs[:5], mode="lsh")
        logger.log_experiment(config, results, sample_pairs)

    return results


# =============================================================================
# Fusion model evaluation
# =============================================================================

def evaluate_fusion(
    model,
    df: pd.DataFrame,
    threshold: float = 0.5,
    sample_size: int = None,
    log_dir: str = None,
    experiment_name: str = "fusion_evaluation",
) -> dict:
    """
    Evaluate the UltimateFusionModel on a labelled dataset.

    Args:
        model: A UltimateFusionModel instance.
        df: DataFrame with 'original' and 'paraphrase' columns.
        threshold: Similarity threshold for binary classification.
        sample_size: Evaluate on a random subsample if given.
        log_dir: If provided, log results via ExperimentLogger.
        experiment_name: Name tag for the logged experiment.

    Returns:
        dict with precision, recall, f1_score, docs_per_second, and
        per-component mean scores.
    """
    eval_df = df.sample(n=sample_size, random_state=42) if sample_size else df
    pairs = list(zip(eval_df["original"], eval_df["paraphrase"]))

    print(f"Evaluating {experiment_name} on {len(pairs)} pairs...")

    all_scores = []
    component_scores = {"semantic": [], "contextual": [], "exact": [], "pattern": []}
    times = []

    for text1, text2 in tqdm(pairs, desc="Processing"):
        t0 = time.time()
        result = model.compute_similarity(text1, text2)
        all_scores.append(result["final_similarity"])
        component_scores["semantic"].append(result["semantic_similarity"])
        component_scores["contextual"].append(result["contextual_similarity"])
        component_scores["exact"].append(result["exact_match_ratio"])
        component_scores["pattern"].append(result["pattern_similarity"])
        times.append(time.time() - t0)

    total_time = sum(times)
    predictions = [1 if s >= threshold else 0 for s in all_scores]

    true_positives = sum(predictions)
    false_negatives = len(predictions) - true_positives
    false_positives = 0

    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "processing_time": total_time,
        "docs_per_second": len(pairs) / total_time if total_time > 0 else 0,
        "total_documents": len(pairs),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "avg_final_similarity": float(np.mean(all_scores)),
        "avg_semantic": float(np.mean(component_scores["semantic"])),
        "avg_contextual": float(np.mean(component_scores["contextual"])),
        "avg_exact": float(np.mean(component_scores["exact"])),
        "avg_pattern": float(np.mean(component_scores["pattern"])),
        "avg_time_ms": float(np.mean(times) * 1000),
    }

    _print_results(experiment_name, results)

    if log_dir:
        logger = ExperimentLogger(log_dir)
        config = {
            "name": experiment_name,
            "params": {
                "threshold": threshold,
                "bert_model": "bert-base-uncased",
                "window_size": model.window_size,
                "ngram_range": model.ngram_range,
                "weights": model.weights,
            },
        }
        sample_pairs = _build_sample_pairs(model, pairs[:5], mode="fusion")
        logger.log_experiment(config, results, sample_pairs)

    return results


# =============================================================================
# Internal helpers
# =============================================================================

def _print_results(name: str, results: dict) -> None:
    print(f"\n{'='*50}")
    print(f"Results: {name}")
    print(f"{'='*50}")
    print(f"  Precision      : {results['precision']:.3f}")
    print(f"  Recall         : {results['recall']:.3f}")
    print(f"  F1 Score       : {results['f1_score']:.3f}")
    print(f"  Docs/sec       : {results['docs_per_second']:.1f}")
    print(f"  Avg latency    : {results['avg_time_ms']:.2f} ms")
    print(f"  Total docs     : {results['total_documents']}")


def _build_sample_pairs(model_or_detector, pairs: list, mode: str = "lsh") -> list:
    samples = []
    for text1, text2 in pairs:
        if mode == "fusion":
            result = model_or_detector.compute_similarity(text1, text2)
        else:
            result = model_or_detector.get_document_similarity(text1, text2)
            if not isinstance(result, dict):
                result = {"final_similarity": float(result)}
        samples.append({
            "original": text1[:200],
            "paraphrase": text2[:200],
            "scores": result,
        })
    return samples
