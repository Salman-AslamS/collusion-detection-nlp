"""
collusion_detection
-------------------
NLP pipeline for academic collusion detection using LSH, MinHash,
and BERT hybrid architectures.

MSc Data Science dissertation project — INM373, City University London.
Dataset: 10,000 document pairs (ltg/en-wiki-paraphrased, HuggingFace).
Experiments: 53 logged runs across 5 model architectures.

Quick start:
    from src import OptimizedLSHDetector, UltimateFusionModel

    # Fast, high-precision LSH detector (932 docs/sec, precision 0.975)
    lsh = OptimizedLSHDetector(threshold=0.35)
    print(lsh.get_document_similarity(text1, text2))

    # BERT + LSH fusion model (best F1, ~5 docs/sec)
    fusion = UltimateFusionModel()
    result = fusion.compute_similarity(text1, text2)
    print(result['final_similarity'])
"""

from .detector_lsh import OptimizedLSHDetector, UltimateLSHDetector
from .detector_fusion import UltimateFusionModel
from .experiment_logger import ExperimentLogger
from .evaluator import evaluate_lsh, evaluate_fusion

__all__ = [
    "OptimizedLSHDetector",
    "UltimateLSHDetector",
    "UltimateFusionModel",
    "ExperimentLogger",
    "evaluate_lsh",
    "evaluate_fusion",
]
