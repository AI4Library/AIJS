# Fairness Audit of Open LLMs for Library Reference Services ⚖

This repository supports our chapter **Equitable Intelligence in Practice: A Fairness Audit of Open Large Language Models for Library Reference Services**.

We evaluate whether **open-weight LLMs**, when prompted as helpful librarians, **systematically vary** their email-style reference responses by **sex** or **race/ethnicity** cues expressed through names. This repo focuses on the study artifacts (data, scripts, figures). For the full, general Fairness Evaluation Protocol (FEP), see https://github.com/AI4Library/FEP.

---

## Main finding at a glance

![Summary of classification margins across models, settings, and demographic dimensions](results/phase1_overview_forest_sig.png)

**Summary:** Across **academic** and **public** library settings, we find **no detectable race/ethnicity-linked differentiation** in responses for any model. Sex-linked signals are generally near chance; the only consistent detectable signal is for **Llama-3.1 (academic setting)**, and it is driven by a **stylistic salutation ("dear")** rather than substantive differences in service quality.

---

## Study design (what we did)

- **Settings**
  - **Academic libraries:** three common reference templates (special collections, sports team name origin, historical population).
  - **Public libraries:** three common templates (readers’ advisory, community information, practical assistance).

- **Identity cues**
  - Users are represented as **email senders** signing with a synthetic English name.
  - Names are sampled to balance **12 groups**: sex (male/female) × race/ethnicity (6 U.S. Census categories).
  - First names come from **SSA baby names**; surnames come from the **2010 U.S. Census surname** table.

- **Models (open)**
  - **Llama-3.1 8B Instruct**
  - **Gemma-2 9B Instruct**
  - **Ministral 8B Instruct**

- **Scale**
  - **5 seeds × 500 interactions per seed** per (model × setting)
  - Target corpus size: **2,500 responses per model per setting** (after filtering generation failures)

---

## Evaluation (high level)

We treat fairness as an empirical question: if responses systematically differ by a protected attribute, a classifier should be able to predict the attribute from response text above chance.

- **Representation:** TF-IDF over top words (reduced for shorter outputs where needed); gendered honorifics masked.
- **Diagnostic classifiers:** Logistic Regression, MLP, XGBoost.
- **Validation:** 5-fold cross-validation (folds correspond to the 5 seeds).
- **Inference:** one-sample tests vs chance with Bonferroni correction.
- **Interpretation:** when a signal is detected, we inspect word-level drivers (volcano plots).

If you want the full protocol framing and reusable implementation details, see https://github.com/AI4Library/FEP.

---

## What’s in this repository

- `run.py`  
  Generate synthetic patron emails and collect model responses for both settings.

- `probe.py`  
  Run the diagnostic classification analysis and produce summary figures.

- `outputs/`  
  Collected prompts and model responses (organized by model, setting, seed).

- `results/`  
  Plots and tables used in the chapter (including the main summary figure above).

---

## Reproduce the results (minimal)

1) Install dependencies
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
````

2. Run the analysis (uses existing outputs)

```bash
python probe.py
```

3. (Optional) Regenerate outputs, then re-run analysis

```bash
python run.py
python probe.py
```

---

## License

MIT License (see `LICENSE`).

---

## Citation

If you use the general protocol beyond this chapter's study, please cite the FEP repository: [https://github.com/AI4Library/FEP](https://github.com/AI4Library/FEP).

```markdown
@article{wang2025fairness,
  title={Fairness Evaluation of Large Language Models in Academic Library Reference Services},
  author={Wang, Haining and Clark, Jason and Yan, Yueru and Bradley, Star and Chen, Ruiyang and Zhang, Yiqiong and Fu, Hengyi and Tian, Zuoyu},
  journal={arXiv preprint arXiv:2507.04224},
  year={2025}
}
```

