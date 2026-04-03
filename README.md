# Collusion Detection NLP

Academic collusion detection pipeline using MinHash LSH and BERT hybrid architectures.

**MSc Data Science dissertation — INM373, City University London**
Dataset: 10,000 document pairs · 53 logged experiments · 5 model architectures

---

## The Problem

Detecting academic collusion is harder than plagiarism detection. Plagiarism involves copying; collusion involves rewriting — paraphrasing, restructuring, synonym substitution. Token matching alone fails. The challenge is distinguishing genuinely independent work from coordinated rewriting at scale.

## Approach

Five models were built in progression, each informed by the previous run's experiment logs:

| Model | Architecture | Precision | Recall | Speed |
|-------|-------------|-----------|--------|-------|
| Model 1 — Base LSH | MinHash + basic preprocessing | 0.132 | 0.089 | 39 docs/sec |
| Model 2 — Linguistic Enhanced | + stopword removal, stemming, lemmatisation | 0.412 | 0.231 | 156 docs/sec |
| Model 3 — OptimizedLSH | + weighted token importance (proper nouns 2.5×, technical terms 2.0×) | **0.975** | 0.687 | **932 docs/sec** |
| Model 4 — UltimateLSH | + context-aware n-gram range (1–3), preserved stopword weighting | 0.961 | 0.743 | 847 docs/sec |
| Model 5 — UltimateFusion | BERT + LSH hybrid fusion | **1.000** | **1.000** | ~5 docs/sec |

**Key finding:** LSH at threshold 0.35 gives near-perfect precision — every flag is correct. Low recall is a deliberate design choice for academic integrity use cases where false accusations carry serious consequences for students. BERT recovers the recall that LSH misses by catching semantic similarity even when token overlap is low.

## Design Decisions

**Why threshold 0.35?** Tested across thresholds 0.25–0.60 in 12 experiments. Below 0.35 precision collapses as LSH bands collide on dissimilar documents. Above 0.40 recall drops sharply. 0.35 is the precision-recall sweet spot for this dataset.

**Why weighted tokens?** Proper nouns (names, places, institutions) are the strongest signal of document relatedness. A student rewriting another's essay will change common words but often preserve named entities. Weighting proper nouns 2.5× and technical terms 2.0× exploits this directly.

**Why BERT for the fusion model?** LSH misses semantic paraphrasing — "the temperature rose significantly" vs "there was a substantial increase in heat." BERT's contextualised embeddings catch meaning-preserving rewrites that token-level methods miss entirely.

**Speed trade-off:** 932 docs/sec (LSH) vs ~5 docs/sec (BERT). In a real university setting processing 500 submissions, LSH completes in under 1 second; the fusion model takes ~100 seconds. Both are acceptable in a batch-processing context run at submission deadline.

## Project Structure

```
collusion_detection/
├── src/
│   ├── __init__.py           # Package with clean imports
│   ├── detector_lsh.py       # OptimizedLSHDetector, UltimateLSHDetector
│   ├── detector_fusion.py    # UltimateFusionModel (BERT + LSH)
│   ├── experiment_logger.py  # Persistent experiment tracking (53 runs)
│   └── evaluator.py          # Evaluation harness for all models
├── notebooks/
│   └── demo.ipynb            # Threshold comparison + live detection demo
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/collusion-detection-nlp.git
cd collusion-detection-nlp
pip install -r requirements.txt
```

## Quick Start

```python
from src import OptimizedLSHDetector, UltimateFusionModel

text1 = "Climate change poses significant risks to coastal ecosystems worldwide."
text2 = "Coastal ecosystems face substantial threats from global climate change."

# Fast LSH detector — best for batch processing
lsh = OptimizedLSHDetector(threshold=0.35)
result = lsh.get_document_similarity(text1, text2)
print(f"LSH similarity: {result['final_similarity']:.3f}")

# BERT fusion model — best accuracy
fusion = UltimateFusionModel()
result = fusion.compute_similarity(text1, text2)
print(f"Fusion similarity: {result['final_similarity']:.3f}")
print(f"  Semantic (BERT): {result['semantic_similarity']:.3f}")
print(f"  Token overlap:   {result['exact_match_ratio']:.3f}")
```

## Indexing and Querying

```python
from src import OptimizedLSHDetector

detector = OptimizedLSHDetector(threshold=0.35)

# Index a corpus
for i, doc in enumerate(documents):
    detector.add_document(f"doc_{i}", doc)

# Query for similar documents
similar_ids = detector.find_similar(query_document)
print(f"Found {len(similar_ids)} similar documents")
```

## Experiment Logging

```python
from src import ExperimentLogger, evaluate_lsh, OptimizedLSHDetector

detector = OptimizedLSHDetector()
results = evaluate_lsh(detector, df, log_dir="logs/", experiment_name="run_54")

logger = ExperimentLogger("logs/")
summary = logger.get_experiment_summary()
print(summary[["experiment_id", "precision", "recall", "f1_score", "documents_per_second"]])
```

## Requirements

```
datasketch>=1.6.5
transformers>=4.46.3
torch>=2.0.0
nltk>=3.9.1
pandas>=2.2.2
numpy>=1.26.4
scikit-learn>=1.5.2
tqdm>=4.66.6
```

## Dataset

[ltg/en-wiki-paraphrased](https://huggingface.co/datasets/ltg/en-wiki-paraphrased) — Wikipedia paragraph pairs with human-verified paraphrase labels. 10,000 pairs used for training and evaluation.

## Academic Context

This project was developed as the dissertation component of the MSc in Data Science at City University London (module INM373). The work was assessed on technical substance; the write-up quality was penalised separately. The system design, experiment methodology, and model progression are the primary contributions.

---

*For questions or collaboration: [your-email@domain.com]*
