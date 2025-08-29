from __future__ import annotations
import logging, pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, TruePositiveRateParity
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

logger = logging.getLogger(__name__)

def to_aif(df: pd.DataFrame, label: str = 'label_bin', protected: str = 'gender'):
    # Keep only required columns and ensure numeric types per AIF360 requirements
    df2 = df[[label, protected]].copy()
    # Encode protected attribute to numeric if needed
    if df2[protected].dtype == object:
        df2[protected] = df2[protected].map({'F': 0.0, 'M': 1.0}).astype(float)
    else:
        df2[protected] = df2[protected].astype(float)
    df2[label] = df2[label].astype(float)
    return BinaryLabelDataset(
        df=df2,
        label_names=[label],
        protected_attribute_names=[protected],
        favorable_label=1,
        unfavorable_label=0,
    )

class BiasAuditor:
    def __init__(self, pipe, X_train, y_train, g_train):
        self.pipe = pipe
        self.X_train, self.y_train, self.g_train = X_train, y_train, g_train
    
    def metric_frame(self, X, y_true, groups) -> MetricFrame:
        y_pred = self.pipe.predict(X)
        return MetricFrame(
            metrics = {"accuracy": accuracy_score,
                       "selection_rate" : selection_rate},
            y_true = y_true, y_pred=y_pred,
            sensitive_features= groups
        )
    def mitigate_reweigh(self, X_train, y_train, groups):
        # Build a minimal frame with label and protected attribute only
        train_df = pd.DataFrame({'y': y_train, 'g': groups})
        aif = to_aif(train_df.rename(columns={'y': 'label_bin', 'g': 'gender'}))
        rw = Reweighing(
            unprivileged_groups=[{'gender': 0.0}],
            privileged_groups=[{'gender': 1.0}],
        ).fit_transform(aif)
        weights = rw.instance_weights.ravel()
        self.pipe.fit(X_train, y_train, logisticregression__sample_weight=weights)
        return self.pipe
    
    def mitigate_expgrad(self, X_train, y_train, groups):
        constraint = DemographicParity()
        # Route sample weights to the final LogisticRegression inside the pipeline
        exp = ExponentiatedGradient(
            self.pipe,
            constraint,
            sample_weight_name="logisticregression__sample_weight",
        )
        # Fairlearn expects X as a 2D container; wrap Series to single-column DataFrame
        X_df = pd.DataFrame({"text": X_train})
        exp.fit(X_df, y_train, sensitive_features=groups)
        self.pipe = exp
        return self.pipe
    
    
    
    
        
    
    
    