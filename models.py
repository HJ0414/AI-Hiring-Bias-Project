from __future__ import annotations
import logging, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

def build_baseline(random_state: int | None = None):
    # Squeeze input to 1D in case upstream passes (n_samples, 1) to satisfy validators
    squeeze = FunctionTransformer(lambda x: getattr(x, 'squeeze', lambda: x)(), validate=False)
    pipe  = make_pipeline(
        squeeze,
        TfidfVectorizer(ngram_range=(1, 2), max_features=50_000),
        LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state)
    )
    return pipe


