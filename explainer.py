from __future__ import annotations
import shap, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def explain(pipe, X: pd.Series, out: Path, background: pd.Series | None = None,
            bg_n: int = 1000, random_state: int | None = None) -> None:
    vectorizer = pipe.named_steps["tfidfvectorizer"]
    model = pipe.named_steps["logisticregression"]

    # 背景集：默认用传入的 background；若没传，用 X 的子样本兜底
    if background is None:
        background = X.sample(n=min(bg_n, len(X)), random_state=random_state)

    X_vec = vectorizer.transform(X)
    X_bg_vec = vectorizer.transform(background)

    explainer = shap.Explainer(model, X_bg_vec)
    shap_values = explainer(X_vec)

    feature_names = vectorizer.get_feature_names_out() if hasattr(vectorizer, "get_feature_names_out") else None
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    

    
    