"""
Module 7 Week B — Core Skills Drill: Pre-Trained Pipelines & Metrics.

Implement the functions below. See the drill guide for full task descriptions.
"""

import os
import re
import string
from collections import Counter 
from transformers import pipeline 
from rouge_score.rouge_scorer import RougeScorer 


# -- Helpers (provided — do NOT modify) --------------------------------------

def _qa_model_name() -> str:
    """Return env override (CI smoke) or the default QA model."""
    return os.environ.get("QA_MODEL_FOR_CI", "distilbert-base-cased-distilled-squad")


def _summ_model_name() -> str:
    """Return env override (CI smoke) or the default summarization model."""
    return os.environ.get("SUMM_MODEL_FOR_CI", "sshleifer/distilbart-cnn-6-6")


# -- Task 1: QA pipeline ------------------------------------------------------

def build_qa_pipeline(model_name: str):
    """
    Construct a Hugging Face question-answering pipeline.

    Returns the pipeline object (callable).
    """
    # build a question-answering pipeline using the given model name (see reading § 3)
    return pipeline("question-answering", model=model_name)


def answer_one(qa, question: str, context: str) -> dict:
    """
    Run the QA pipeline on one (question, context) pair.

    Returns the pipeline output dict with keys "answer", "score", "start", "end".
    """
    # call qa(question=..., context=...) and return the result
    return qa(question=question, context=context)


# -- Task 2: Normalization + EM ----------------------------------------------

def normalize_answer(s: str) -> str:
    """
    SQuAD-style normalization.

    Apply (in order):
      - lowercase
      - strip standalone articles (a, an, the) using word-boundary regex
      - strip all string.punctuation
      - collapse whitespace
    """
    # apply the four normalization steps in order; word-boundary regex is required for the article strip
 # 1. Lowercase
    s = s.lower()
    # 2. Strip articles (a, an, the) using word-boundary regex 
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # 3. Strip all string.punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # 4. Collapse whitespace
    s = " ".join(s.split())
    return s

def exact_match(pred: str, gold: str) -> int:
    """
    Return 1 if normalized prediction equals normalized gold, else 0.
    """
    # normalize both, compare, return int
    return 1 if normalize_answer(pred) == normalize_answer(gold) else 0


# -- Task 3: Token-F1 --------------------------------------------------------

def token_f1(pred: str, gold: str) -> float:
    """
    Compute token-F1 between prediction and gold after normalization.

    Empty handling:
      - both empty (after normalization) -> 1.0
      - one empty -> 0.0
    Returns a float in [0.0, 1.0]. Never returns NaN.
    """
    # normalize both, split on whitespace
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    # handle empty cases
    if len(pred_tokens) == 0 and len(gold_tokens) == 0: return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0: return 0.0

    #  compute multiset overlap, precision, recall, harmonic mean
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0: return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

# -- Task 4: Summarization pipeline ------------------------------------------

def build_summarizer(model_name: str):
    """
    Construct a Hugging Face summarization pipeline.

    Returns the pipeline object (callable).
    """
    # build a summarization pipeline using the given model name (see reading § 6)
    return pipeline("summarization", model=model_name)


def summarize_one(summ, text: str, max_length: int, min_length: int) -> str:
    """
    Run the summarization pipeline on one document.

    Use do_sample=False, num_beams=4. Return the summary_text string from the
    first output element (the pipeline returns a list-of-dicts).
    """
    # invoke the pipeline with deterministic generation parameters and return the summary string
    #       (the pipeline returns a list-of-dicts — see reading § 6 for the output shape)
    results = summ(
        text, 
        max_length=max_length, 
        min_length=min_length, 
        do_sample=False, 
        num_beams=4
    )
    return results[0]["summary_text"]

# -- Task 5: ROUGE -----------------------------------------------------------

def compute_rouge(pred: str, ref: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 between predicted and reference.

    Use rouge_score.rouge_scorer.RougeScorer with use_stemmer=True.
    Argument order is scorer.score(reference, predicted) — reference FIRST.

    Returns {"rouge1": float, "rouge2": float, "rougeL": float}, all F1.
    """
    # build a stemming-enabled ROUGE scorer over the three metric variants
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    scores = scorer.score(ref, pred)

    # score the (reference, predicted) pair (mind the argument order) and return F1 measures only
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }

if __name__ == "__main__":
    # Minimal smoke when run directly: tasks 2/3/5 don't need network.
    sample_norm = normalize_answer("The Eiffel Tower")
    print("normalize_answer('The Eiffel Tower') ->", repr(sample_norm))
    print("exact_match('The Eiffel Tower', 'eiffel tower') ->", exact_match("The Eiffel Tower", "eiffel tower"))
    print("token_f1('the cat sat on the mat', 'cat sat on mat') ->", token_f1("the cat sat on the mat", "cat sat on mat"))
    print("compute_rouge('a b c d', 'a b c d') ->", compute_rouge("a b c d", "a b c d"))
