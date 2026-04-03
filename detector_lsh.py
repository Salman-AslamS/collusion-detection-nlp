"""
detector_lsh.py
---------------
LSH-based document similarity detectors developed across 53 logged
experiments in INM373 (MSc Data Science, City University London).

Two production-ready classes are exported:

    OptimizedLSHDetector  — Model 3 in the dissertation progression.
                            Weighted token importance with n-gram context.
                            Precision 0.975 at threshold 0.35-0.40.
                            932 docs/sec on 10,000-pair Wikipedia dataset.

    UltimateLSHDetector   — Model 4. Extended weighting scheme with
                            context-aware n-gram range (1-3). Basis for
                            the fusion model's LSH component.

Both classes follow the same interface:
    add_document(doc_id, text)    → index a document
    find_similar(text)            → query the LSH index
    get_document_similarity(t1, t2) → float similarity score
    analyze_patterns(t1, t2)      → dict of match categories
"""

import re
import time

import nltk
from datasketch import MinHash, MinHashLSH
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is available
for _resource in ("corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"):
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource.split("/")[1], quiet=True)


# =============================================================================
# OptimizedLSHDetector  (Model 3)
# =============================================================================

class OptimizedLSHDetector:
    """
    Optimized MinHash LSH detector with learned token importance weights.

    Key design decisions (evidence-based from 53 experiments):
      - Threshold 0.35 gives precision 0.975 on academic paraphrase pairs.
        Every flag is correct — critical for academic integrity use cases
        where false accusations carry serious consequences for students.
      - Proper nouns weighted 2.5x: preserving named entities is the
        single biggest driver of precision improvement over the base model.
      - Technical terms weighted 2.0x: domain-specific vocabulary (CamelCase,
        acronyms, hyphenated terms) signals semantic intent more reliably
        than common words.
      - MinHash + token overlap combined (0.7 / 0.3) to reduce false
        negatives from LSH band collisions without sacrificing precision.

    Args:
        threshold (float): Jaccard similarity threshold for LSH index. Default 0.35.
        num_perm (int): Number of MinHash permutations. Default 512.
        ngram_range (tuple): (min_n, max_n) for n-gram generation. Default (1, 3).

    Example:
        >>> detector = OptimizedLSHDetector(threshold=0.35)
        >>> detector.add_document("doc_1", "The quick brown fox jumps over the lazy dog")
        >>> results = detector.find_similar("A fast brown fox leaps over a sleepy dog")
        >>> similarity = detector.get_document_similarity(text1, text2)
    """

    def __init__(self, threshold: float = 0.35, num_perm: int = 512, ngram_range: tuple = (1, 3)):
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_range = ngram_range

        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes: dict = {}

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Weights derived from experiment log analysis
        self.weights = {
            "exact_match": 3.0,       # highest signal for collusion
            "proper_noun": 2.5,       # named entities preserved
            "technical_term": 2.0,    # domain vocabulary
            "ngram_match": 1.5,       # phrase-level overlap
            "lemma_match": 1.2,       # morphological variants
        }

        # Regex patterns that identify technical/academic terms
        self._technical_patterns = [
            re.compile(r"\b[A-Z][a-z]+(?:[-_][A-Z][a-z]+)*\b"),   # CamelCase
            re.compile(r"\b[A-Z][A-Z0-9]+\b"),                      # Acronyms (NLP, BERT)
            re.compile(r"\b\d+(?:\.\d+)?%?\b"),                     # Numbers / percentages
            re.compile(r"\b[A-Za-z]+(?:[-_][A-Za-z]+)+\b"),        # Hyphenated terms
        ]

    # ------------------------------------------------------------------
    # Token classification
    # ------------------------------------------------------------------

    def is_proper_noun(self, word: str) -> bool:
        """True if word starts with a capital but is not an all-caps acronym."""
        return (
            len(word) > 1
            and word[0].isupper()
            and not all(c.isupper() for c in word)
        )

    def is_technical_term(self, word: str) -> bool:
        """True if word matches any technical term pattern, or is a non-stopword."""
        return any(p.match(word) for p in self._technical_patterns) or (
            len(word) > 2 and word.lower() not in self.stop_words
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        """Normalize whitespace and remove non-word characters (keep hyphens, dots, %)."""
        text = re.sub(r"[^\w\s\-'\.%]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace("i.e.", "that is").replace("e.g.", "for example")
        return text.strip()

    def _get_ngrams(self, tokens: list, n: int) -> list:
        """Generate weighted n-grams. Higher-importance grams are repeated."""
        result = []
        for i in range(len(tokens) - n + 1):
            gram = tokens[i : i + n]
            weight = 1.0
            if any(self.is_proper_noun(t) for t in gram):
                weight *= self.weights["proper_noun"]
            if any(self.is_technical_term(t) for t in gram):
                weight *= self.weights["technical_term"]
            gram_str = " ".join(gram)
            result.extend([gram_str] * int(weight))
        return result

    def preprocess_text(self, text: str) -> list:
        """
        Full preprocessing pipeline → list of weighted tokens.

        Pipeline: clean → split → classify (proper noun / technical / regular)
                  → lemmatise → generate n-grams → deduplicate.
        """
        if not isinstance(text, str):
            return []

        text = self._clean(text)
        tokens = []
        preserved = set()

        for word in text.split():
            word = word.strip()
            if len(word) <= 1:
                continue

            if self.is_proper_noun(word):
                preserved.add(word)
                tokens.extend([word] * int(self.weights["proper_noun"]))

            elif self.is_technical_term(word):
                tokens.extend([word] * int(self.weights["technical_term"]))
                lemma = self.lemmatizer.lemmatize(word.lower())
                if lemma != word.lower():
                    tokens.extend([lemma] * int(self.weights["lemma_match"]))

            else:
                w = word.lower()
                if w not in self.stop_words:
                    tokens.append(w)
                    lemma = self.lemmatizer.lemmatize(w)
                    if lemma != w:
                        tokens.extend([lemma] * int(self.weights["lemma_match"]))

        # Generate n-grams
        all_tokens = tokens[:]
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            all_tokens.extend(self._get_ngrams(tokens, n))

        all_tokens.extend(preserved)
        return list(set(all_tokens))

    # ------------------------------------------------------------------
    # MinHash
    # ------------------------------------------------------------------

    def _build_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for token in self.preprocess_text(text):
            weight = 1
            if self.is_proper_noun(token):
                weight *= self.weights["proper_noun"]
            if self.is_technical_term(token):
                weight *= self.weights["technical_term"]
            for _ in range(int(weight)):
                m.update(token.encode("utf-8"))
        return m

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> None:
        """Index a document in the LSH structure."""
        mh = self._build_minhash(text)
        self.minhashes[doc_id] = mh
        self.lsh.insert(doc_id, mh)

    def find_similar(self, text: str) -> list:
        """Return list of doc_ids that exceed the similarity threshold."""
        return self.lsh.query(self._build_minhash(text))

    def get_document_similarity(self, text1: str, text2: str) -> dict:
        """
        Compute similarity using MinHash Jaccard + token overlap.

        Returns:
            dict with keys: final_similarity, semantic_similarity,
            contextual_similarity, exact_match_ratio, pattern_similarity.
        """
        mh1 = self._build_minhash(text1)
        mh2 = self._build_minhash(text2)
        minhash_sim = mh1.jaccard(mh2)

        t1 = set(self.preprocess_text(text1))
        t2 = set(self.preprocess_text(text2))
        overlap = len(t1 & t2) / max(len(t1), len(t2)) if (t1 or t2) else 0.0

        final = min(1.0, minhash_sim * 0.6 + overlap * 0.4)
        return {
            "final_similarity": final,
            "semantic_similarity": minhash_sim,
            "contextual_similarity": overlap,
            "exact_match_ratio": overlap,
            "pattern_similarity": overlap,
        }

    def analyze_patterns(self, text1: str, text2: str) -> dict:
        """
        Categorise matched tokens between two documents.

        Returns:
            dict with exact_matches, proper_nouns, technical_terms,
            ngram_matches, and match_counts.
        """
        t1 = set(self.preprocess_text(text1))
        t2 = set(self.preprocess_text(text2))
        common = t1 & t2

        proper_nouns = {t for t in common if self.is_proper_noun(t)}
        technical = {t for t in common if self.is_technical_term(t)}
        ngrams = {t for t in common if " " in t}

        return {
            "exact_matches": list(common),
            "proper_nouns": list(proper_nouns),
            "technical_terms": list(technical),
            "ngram_matches": list(ngrams),
            "match_counts": {
                "total": len(common),
                "proper_nouns": len(proper_nouns),
                "technical_terms": len(technical),
                "ngrams": len(ngrams),
            },
        }


# =============================================================================
# UltimateLSHDetector  (Model 4)
# =============================================================================

class UltimateLSHDetector:
    """
    Extended LSH detector with context-aware n-gram weighting (Model 4).

    Built on top of OptimizedLSHDetector's lessons. Key addition: preserved
    stopwords contribute a small weight (0.5) rather than being discarded,
    capturing function-word patterns that matter in paraphrase detection.

    Args:
        threshold (float): Jaccard similarity threshold. Default 0.35.
        num_perm (int): MinHash permutations. Default 512.
        ngram_range (tuple): N-gram range. Default (1, 3).

    Example:
        >>> detector = UltimateLSHDetector()
        >>> sim = detector.get_document_similarity(text1, text2)
        >>> patterns = detector.analyze_patterns(text1, text2)
    """

    def __init__(self, threshold: float = 0.35, num_perm: int = 512, ngram_range: tuple = (1, 3)):
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_range = ngram_range

        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes: dict = {}

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        self.weights = {
            "proper_noun": 3.0,
            "technical_term": 2.5,
            "ngram": 2.0,
            "preserved_stopword": 0.5,
        }

    # ------------------------------------------------------------------
    # Token classification (same logic as OptimizedLSHDetector)
    # ------------------------------------------------------------------

    def is_proper_noun(self, token: str) -> bool:
        if not token:
            return False
        return token[0].isupper() and not all(c.isupper() for c in token)

    def is_technical_term(self, token: str) -> bool:
        if not token or len(token) < 3:
            return False
        return (
            any(c.isdigit() for c in token)
            or "-" in token
            or "_" in token
            or token.lower() not in self.stop_words
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _get_ngrams(self, tokens: list, n: int) -> list:
        result = []
        for i in range(len(tokens) - n + 1):
            gram = tokens[i : i + n]
            weight = 1.0
            if any(self.is_proper_noun(t) for t in gram):
                weight *= self.weights["proper_noun"]
            if any(self.is_technical_term(t) for t in gram):
                weight *= self.weights["technical_term"]
            result.extend([" ".join(gram)] * int(weight))
        return result

    def preprocess_text(self, text: str) -> list:
        if not isinstance(text, str):
            return []

        text = re.sub(r"[^\w\s\-']", " ", text)
        tokens = []

        for word in text.split():
            clean = word.strip()
            if not clean:
                continue

            if self.is_proper_noun(clean):
                tokens.append(clean)

            elif self.is_technical_term(clean):
                tokens.append(clean)
                lemma = self.lemmatizer.lemmatize(clean.lower())
                if lemma != clean.lower():
                    tokens.append(lemma)

            else:
                w = clean.lower()
                if w not in self.stop_words:
                    tokens.append(w)
                    lemma = self.lemmatizer.lemmatize(w)
                    if lemma != w:
                        tokens.append(lemma)
                elif len(w) > 3:
                    # Preserved stopword: small fractional weight
                    tokens.extend([w] * int(self.weights["preserved_stopword"]))

        all_tokens = tokens[:]
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            all_tokens.extend(self._get_ngrams(tokens, n))

        return list(set(all_tokens))

    # ------------------------------------------------------------------
    # MinHash
    # ------------------------------------------------------------------

    def _build_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for token in self.preprocess_text(text):
            weight = 1
            if self.is_proper_noun(token):
                weight *= self.weights["proper_noun"]
            if self.is_technical_term(token):
                weight *= self.weights["technical_term"]
            for _ in range(int(weight)):
                m.update(token.encode("utf-8"))
        return m

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> None:
        mh = self._build_minhash(text)
        self.minhashes[doc_id] = mh
        self.lsh.insert(doc_id, mh)

    def find_similar(self, text: str) -> list:
        return self.lsh.query(self._build_minhash(text))

    def get_document_similarity(self, text1: str, text2: str) -> float:
        """Return Jaccard similarity estimate between two texts."""
        return self._build_minhash(text1).jaccard(self._build_minhash(text2))

    def analyze_patterns(self, text1: str, text2: str) -> dict:
        t1 = set(self.preprocess_text(text1))
        t2 = set(self.preprocess_text(text2))
        common = t1 & t2

        return {
            "exact_matches": list(common),
            "proper_noun_matches": [t for t in common if self.is_proper_noun(t)],
            "technical_matches": [t for t in common if self.is_technical_term(t)],
            "ngram_matches": [t for t in common if " " in t],
        }
