# AI Bias Audit

## Overview
This project trains a simple text classifier (tech vs non‑tech profession) and audits fairness across gender. It demonstrates:
- Data generation (synthetic BIOS-like dataset if missing)
- Text preprocessing (cleaning + stopwords)
- Baseline model: TF‑IDF + Logistic Regression
- Fairness metrics by group (using Fairlearn)
- Mitigations: Reweighing (AIF360, optional) and Exponentiated Gradient (Fairlearn)
- Model explanations with SHAP (summary bar plot)

## Setup
```bash
conda env create -f environment.yml
conda activate ai-bias-audit
```
If SHAP complains about plotting, ensure `matplotlib` is installed (it is listed in the env or install manually):
```bash
conda install matplotlib -y
```

## Run
```bash
python cli.py run
```
Optional config path (defaults to `configs/default.yaml`):
```bash
python cli.py run --cfg-path configs/default.yaml
```

## Configuration
File: `configs/default.yaml`
- `sample_size`: number of rows to sample from the dataset
- `random_seed`: seed for reproducibility
- `mitigation`: list, any of ["reweighing", "exponentiated_gradient"]
- `data_dir`: input data directory (a synthetic `bios_bias.csv` will be created if missing)
- `out_dir`: output directory for artefacts

## Outputs
- `outputs/baseline_metrics.csv`: by‑group metrics for the baseline
- `outputs/rw_metrics.csv`: by‑group metrics with Reweighing (if enabled and AIF360 installed)
- `outputs/exp_metrics.csv`: by‑group metrics with Exponentiated Gradient
- `outputs/shap_summary.png`: SHAP summary bar plot of most important features

## Project Structure
```
AI-Bias-Audit/
├─ cli.py              # Entry point (Typer CLI)
├─ audit.py            # BiasAuditor (metrics + mitigations)
├─ model.py            # Baseline pipeline builder
├─ explainer.py        # SHAP explanation utilities
├─ data.py             # DatasetManager (creates synthetic data if absent)
├─ config.py           # Config loading (pydantic)
├─ configs/            # YAML configs (default.yaml)
├─ data/               # Data directory (bios_bias.csv)
├─ outputs/            # Outputs (metrics, plots)
├─ environment.yml     # Conda environment
├─ README.md           # This file
└─ .gitignore
```

## Notes & Troubleshooting
- Stopwords: the first run may auto‑download NLTK stopwords; if offline, a small built‑in set is used as fallback.
- AIF360: if installation is difficult on macOS, you can disable `reweighing` in config. The baseline and `exponentiated_gradient` path will still work.
- Reproducibility: the random seed is applied to sampling, dataset generation, and model (via `build_baseline(random_state=...)`).

## License
See `LICENSE`.
