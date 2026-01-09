
import math
import re
import warnings
import os
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from detoxify import Detoxify
from presidio_analyzer import AnalyzerEngine
import torch
from bert_score import score as bert_score
from transformers import logging as transformers_logging

# Suppress transformers warnings about model weights initialization
transformers_logging.set_verbosity_error()
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

_EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_VADER = SentimentIntensityAnalyzer()
_ANALYZER = AnalyzerEngine()

def semantic_parity_score(text_a: str, text_b: str) -> float:
    """
    Calculate semantic parity score using BERT-score.
    
    BERT-score uses contextual embeddings to compute token-level similarity,
    providing a more accurate measure of semantic similarity than cosine similarity
    on sentence embeddings.
    
    Args:
        text_a: First text to compare
        text_b: Second text to compare
        
    Returns:
        F1 score from BERT-score (ranges from 0 to 1, higher is more similar)
    """
    # Handle empty strings
    if not text_a.strip() or not text_b.strip():
        return 0.0
    
    try:
        # Suppress warnings during BERT-score computation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            # BERT-score returns (precision, recall, F1) tensors
            # We use F1 score as it balances precision and recall
            # lang='en' specifies English, model_type='roberta-large' is used by default
            P, R, F1 = bert_score(
                [text_a],
                [text_b],
                lang='en',
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                batch_size=1  # Process one at a time to avoid memory issues
            )
        # F1 is a tensor, convert to float
        return float(F1[0].item())
    except Exception as e:
        # Fallback to embedding-based cosine similarity if BERT-score fails
        print(f"Warning: BERT-score failed ({e}), falling back to embedding similarity")
    ea = _EMB_MODEL.encode(text_a, normalize_embeddings=True)
    eb = _EMB_MODEL.encode(text_b, normalize_embeddings=True)
    return float(util.cos_sim(ea, eb)[0][0])

_POS = set(["good","great","helpful","clear","easy","fast","accurate","approved"])
_NEG = set(["bad","poor","confusing","hard","slow","inaccurate","denied","rejected"])


def sentiment_polarity_gap(text_a, text_b):
    va = _VADER.polarity_scores(text_a)['compound']
    vb = _VADER.polarity_scores(text_b)['compound']
    return abs(va - vb)

def bias_lexicon_frequency(text, sensitive_terms):
    check_terms = text.split()
    check_terms = set(t.lower() for t in check_terms)
    sensitive_terms = set(t.lower() for t in sensitive_terms)
    if not check_terms and not sensitive_terms:
        return 0.0
    inter = len(check_terms & sensitive_terms)
    union = len(check_terms | sensitive_terms)
    return round(inter/union if union else 0.0, 4)

def bias_lexicon_frequency_gap(text_a, text_b, sensitive_terms):
    """Calculate the difference in bias lexicon frequency between two texts."""
    blf_a = bias_lexicon_frequency(text_a, sensitive_terms)
    blf_b = bias_lexicon_frequency(text_b, sensitive_terms)
    return abs(blf_a - blf_b)
  
def language_equivalence_score(text_a, text_b):
    return semantic_parity_score(text_a, text_b)

def retrieval_fairness_index(doc_ids_a, doc_ids_b, k=5):
    a = set(doc_ids_a[:k])
    b = set(doc_ids_b[:k])
    if not a and not b: 
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return round(inter/union if union else 0.0, 4)

def toxicity_score(text):
    model = Detoxify('original')
    return model.predict(text)['toxicity']
    
def citation_completeness(citations, min_required):
    count = len(citations or [])
    return (count >= int(min_required)), count

def doc_recency_index(retrieved, active_year):
    if not retrieved:
        return 0.0
    horizon = 3
    recent = 0
    for d in (retrieved or []):
        y = d.get("year")
        try:
            y = int(y)
        except Exception:
            y = None
        if y is not None and (active_year - y) <= horizon:
            recent += 1
    return round(recent / max(1, len(retrieved)), 4)

def authority_score(retrieved, weights):
    if not retrieved:
        return 0.0
    wsum = 0.0
    for d in (retrieved or []):
        st = d.get("source_type", "unknown")
        wsum += float(weights.get(st, weights.get("unknown", 0.3)))
    return round(wsum / max(1, len(retrieved)), 4)

def validate_no_pii_in_text(text: str) -> tuple[bool, list]:
    """
    Validate that text doesn't contain PII data.
    
    Args:
        text: Text to validate
        
    Returns:
        tuple: (is_clean, detected_entities) where is_clean is True if no PII detected
    """
    if not text or len(text.strip()) == 0:
        return True, []
    
    results = _ANALYZER.analyze(text=text, language="en")
    detected_entities = [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": r.score,
            "text": text[r.start:r.end] if r.start < len(text) and r.end <= len(text) else ""
        }
        for r in results
    ]
    
    # Check if any PII entities were detected
    is_clean = len(results) == 0
    
    return is_clean, detected_entities

