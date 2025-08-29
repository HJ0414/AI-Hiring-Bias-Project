import logging, typer
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split
import pandas as pd
from config import load_config
from data import DatasetManager
from models import build_baseline
from audit import BiasAuditor
from explainer import explain

app = typer.Typer()
logging.basicConfig(level = logging.INFO, handlers = [RichHandler()], format = "%(message)s")

@app.command()
def run(cfg_path: str = "configs/default.yaml"):
    cfg = load_config(cfg_path)
    dm = DatasetManager(cfg); dm.fetch_all()
    cfg.out_dir.mkdir(exist_ok = True, parents = True)
    df = pd.read_csv(cfg.data_dir / "bios_bias.csv")
    df = df.sample(n=min(cfg.sample_size, len(df)), random_state=cfg.random_seed)
    try:
        from nltk.corpus import stopwords
        try:
            STOP = set(stopwords.words("english"))
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            STOP = set(stopwords.words("english"))
    except Exception:
        STOP = {"the","a","an","and","or","to","of","in","for","on","with","is","are","was","were"}

    df['clean'] = (df ['bio'].astype(str)
                   .str.replace(r"https?://\S+", " ", regex = True)
                   .str.replace(r"[^A-Za-z\s]", " ", regex=True)
                   .str.lower().str.split()
                   .apply(lambda toks: " ".join(t for t in toks if t not in STOP)))
    tech = {"software engineer", "data scientist", "system administrator", "security analyst"}
    df["label_bin"] = df["profession"].isin(tech).astype(int)

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        df["clean"], df["label_bin"], df["gender"].map({0: "F", 1: "M"}),
        test_size=0.2, random_state=cfg.random_seed, stratify=df["label_bin"]
    )
    pipe = build_baseline(random_state = cfg.random_seed);pipe.fit(X_train, y_train)
    auditor = BiasAuditor(pipe, X_train, y_train, g_train)
    base_metrics = auditor.metric_frame(X_test, y_test, g_test)
    base_metrics.by_group.to_csv(cfg.out_dir/"baseline_metrics.csv")

    if "reweighing" in cfg.mitigation:
        pipe_rw = auditor.mitigate_reweigh(X_train, y_train, g_train)
        rw_metrics = auditor.metric_frame(X_test, y_test, g_test)
        rw_metrics.by_group.to_csv(cfg.out_dir/"rw_metrics.csv")
    if "exponentiated_gradient" in cfg.mitigation:
        pipe_exp = auditor.mitigate_expgrad(X_train, y_train, g_train)
        exp_metrics = auditor.metric_frame(X_test, y_test, g_test)
        exp_metrics.by_group.to_csv(cfg.out_dir/"exp_metrics.csv")
    
    explain(auditor.pipe.estimator, X_test.iloc[:100], cfg.out_dir/"shap_summary.png")
    logging.info("🎉 All done. See outputs/ for artefacts.")

if __name__ == "__main__":
    app()
    


    




