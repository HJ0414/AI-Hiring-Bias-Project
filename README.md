# AI-Hiring-Bias-Project
# AI Hiring Bias Audit

## What is this?
A project to audit and mitigate gender bias in AI résumé screening using public datasets and machine learning, following a reproducible 20-day game plan.

## Main Features
- End-to-end pipeline: download, preprocess, model, audit, debias, explain.
- Uses TF-IDF + Logistic Regression as the baseline.
- Measures gender bias and applies two mitigation techniques (Reweighing, Adversarial Debiasing).
- Optional Streamlit demo app for résumé prediction and fairness disclaimer.

## Try it out
1. **Set up the environment:**
conda env create -f environment.yml
conda activate ai-bias-audit

2. **Run the pipeline:**
3. **(Optional) Launch demo:**
4. ## Data used
- [Bias-in-Bios](https://github.com/microsoft/biosbias)
- [Resume Screening Bias Corpus](https://github.com/kyrawilson/Resume-Screening-Bias)
- [SSA First Names](https://www.ssa.gov/oact/babynames/names.zip)
- [Census Surnames](https://www2.census.gov/topics/genealogy/2010surnames/names.zip)

## Results
- Fairness metrics and plots in `/outputs`
- SHAP explainability
- Detailed plan in [Ultimate 20-Day-AI-Hiring-Bias-Audit-Project.pdf]

## License
MIT

## Limitations
Current data is binary gender only due to public dataset constraints. See [plan] for details/future work.


