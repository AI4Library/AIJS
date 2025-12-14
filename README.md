# Fairness Audit of Open LLMs for Library Reference Services ⚖

This repository supports our chapter **Equitable Intelligence in Practice: A Fairness Audit of Open Large Language Models for Library Reference Services**.

This chapter is our contribution to the Call for Abstracts for **Equitable Intelligence: Artificial Intelligence and Social Justice Intersections in Library and Information Science**, edited by **Hengyi Fu, Souvick (Vic) Ghosh, Darra Hofman, and Bharat Mehra**.

We evaluate whether **open-weight LLMs**, when prompted as helpful librarians, **systematically vary** their email-style reference responses by **sex** or **race/ethnicity** cues expressed through names. This repo focuses on the study artifacts (data, scripts, figures). For the full, general Fairness Evaluation Protocol (FEP), see https://github.com/AI4Library/FEP.

---

## Main finding

![Summary of classification margins across models, settings, and demographic dimensions](phase1_overview_forest_sig.png)

**Summary:** Across **academic** and **public** library settings, we find **no detectable race/ethnicity-linked differentiation** in responses for any model. Sex-linked signals are generally near chance; the only consistent detectable signal is for **Llama-3.1 (academic setting)**, and it is driven by a **stylistic salutation ("dear")** rather than substantive differences in service quality.

---

## Study design

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

## Evaluation

We treat fairness as an empirical question: if responses systematically differ by a protected attribute, a classifier should be able to predict the attribute from response text above chance.

- **Representation:** TF-IDF over top words (reduced for shorter outputs where needed); gendered honorifics masked.
- **Diagnostic classifiers:** Logistic Regression, MLP, XGBoost.
- **Validation:** 5-fold cross-validation (folds correspond to the 5 seeds).
- **Inference:** one-sample tests vs chance with Bonferroni correction.
- **Interpretation:** when a signal is detected, we inspect word-level drivers (volcano plots).

If you want the full protocol framing and reusable implementation details, see https://github.com/AI4Library/FEP.

---

## License

MIT

---

## Contributing

Contributions are welcome! Please open an issue before submitting a pull request.

## Contact

- **Reproduction / data / code questions:** Haining Wang (hw56@iu.edu)
- **General questions / project context:** Jason Clark (jaclark@montana.edu)



[//]: # (---)

[//]: # ()
[//]: # (## Citation)

[//]: # ()
[//]: # (todo)

[//]: # (```markdown)

[//]: # (todo)

[//]: # (```)
