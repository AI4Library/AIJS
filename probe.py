"""
Bias probing via attribute classification of LLM outputs (open models only)

This script probes whether open-weight model outputs exhibit systematic variation across
demographic characteristics (sex, race/ethnicity, patron type).

it loads seed-wise JSON files produced by:
- academic: run.py  -> academic_outputs/
- public:   public_run.py -> public_outputs/

then it trains simple text classifiers to predict demographic labels from response text.
we use content words only:
- tf-idf features with english stopwords removed
- a fixed tf-idf vocabulary size (50 terms) across all models and domains to keep runs
  comparable and to reduce statsmodels instability

evaluation:
- leave-one-seed-out splits (one seed held out per fold)

models:
- open models only: llama-3.1-8b, ministral-8b, gemma-2-9b

outputs:
- probe.json (default) with results for the requested domains
- probe_ablation.json if you run --ablation (llama temp 0.0 vs 0.3, academic only)

usage:
python probe.py
python probe.py --domains academic
python probe.py --domains public
python probe.py --domains academic public --output_path probe.json
python probe.py --ablation
python probe.py --debug
"""

import argparse
import json
import os
import string
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

# open models only (strictly comparable across academic/public)
OPEN_MODEL_NAMES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "google/gemma-2-9b-it",
]

# default output dirs in your repo layout
DOMAIN_TO_DIR = {
    "academic": "academic_outputs",
    "public": "public_outputs",
}

# fixed tf-idf vocabulary size (same for all models and domains)
TFIDF_MAX_FEATURES = 50


def load_data(
    model_name: str,
    characteristic: str,
    input_dir: str,
    failure_token: str = "[NO_TEXT_AFTER_RETRIES]",
    *,
    temperature_filter: float | None = None,
) -> pd.DataFrame:
    """
    load model outputs and return a df with columns: response, label, seed

    temperature_filter:
      - None: load files like <tag>_seed_*.json
      - float: load files like <tag>_temp{temperature}_seed_*.json
    """
    assert characteristic in ["sex", "race_ethnicity", "patron_type"], (
        "characteristic must be one of: sex, race_ethnicity, patron_type"
    )

    tag = model_name.split("/")[-1].replace("-", "_").replace("/", "_")

    if temperature_filter is None:
        prefix = f"{tag}_seed_"
    else:
        prefix = f"{tag}_temp{temperature_filter}_seed_"

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    files = [
        f for f in os.listdir(input_dir) if f.startswith(prefix) and f.endswith(".json")
    ]
    if not files:
        raise FileNotFoundError(
            f"no matching files in {input_dir} for prefix='{prefix}' (model={model_name})"
        )

    rows = []
    for file in files:
        path = os.path.join(input_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or []
        for entry in data:
            response = entry.get("response", "")
            if not response:
                continue
            if failure_token in response:
                continue
            rows.append(
                {
                    "response": response,
                    "label": entry.get(characteristic, None),
                    "seed": entry.get("seed", None),
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["response", "label", "seed"]).reset_index(drop=True)
    df["seed"] = df["seed"].astype(int)
    return df


def compute_ci(accs, confidence=0.95):
    mean = float(np.mean(accs))
    sem = float(np.std(accs, ddof=1) / np.sqrt(len(accs)))
    h = float(sem * t.ppf((1 + confidence) / 2.0, len(accs) - 1))
    return mean, (mean - h, mean + h)


def get_feature_weights(clf, feature_names, model_type):
    if model_type == "logistic":
        weights = clf.coef_[0]
        return (
            pd.DataFrame({"feature": feature_names, "weight": weights})
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )

    if model_type == "mlp":
        weights = clf.coefs_[0][:, 0]
        return (
            pd.DataFrame({"feature": feature_names, "weight": weights})
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )

    if model_type == "xgboost":
        booster = clf.get_booster()
        importance = booster.get_score(importance_type="weight")
        df = pd.DataFrame(
            {"feature": list(importance.keys()), "weight": list(importance.values())}
        ).sort_values("weight", ascending=False)
        return df.reset_index(drop=True)

    raise ValueError(f"unsupported model type: {model_type}")


def build_content_vectorizer(max_features: int):
    stop_words_set = set(ENGLISH_STOP_WORDS).union({"mr", "ms", "mrs", "miss"})

    class ContentTokenizer:
        def __init__(self):
            self.exclusion_set = stop_words_set

        def __call__(self, doc):
            tokens = [t.strip(string.punctuation).lower() for t in doc.split()]
            return [t for t in tokens if t and t not in self.exclusion_set]

    return TfidfVectorizer(
        tokenizer=ContentTokenizer(),
        token_pattern=None,
        max_features=max_features,
    )


def encode_labels_for_statsmodels(df: pd.DataFrame):
    """
    statsmodels MNLogit uses the last class as baseline; we set reference groups last.
    for binary sex, we encode F=0, M=1 so statsmodels Logit models P(M).
    """
    labels = set(df["label"].unique())

    if labels == {"F", "M"}:
        classes = np.array(["F", "M"])
        label_kind = "sex"
    elif labels == {
        "White",
        "Black or African American",
        "Asian or Pacific Islander",
        "American Indian or Alaska Native",
        "Two or More Races",
        "Hispanic or Latino",
    }:
        classes = np.array(
            [
                "Black or African American",
                "Asian or Pacific Islander",
                "American Indian or Alaska Native",
                "Two or More Races",
                "Hispanic or Latino",
                "White",
            ]
        )
        label_kind = "race_ethnicity"
    elif labels == {
        "Undergraduate student",
        "Faculty",
        "Graduate student",
        "Alumni",
        "Staff",
        "Outside user",
    }:
        classes = np.array(
            [
                "Graduate student",
                "Faculty",
                "Staff",
                "Alumni",
                "Outside user",
                "Undergraduate student",
            ]
        )
        label_kind = "patron_type"
    else:
        raise RuntimeError(
            f"label mismatch: unexpected label set encountered:\n{sorted(labels)}"
        )

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(class_to_idx).astype(int).to_numpy()
    return y, classes, label_kind


def safe_get_pvalues(sm_results, n_params: int):
    """
    statsmodels can fail to compute p-values if the hessian/covariance is singular.
    in that case we return NaNs instead of raising.
    """
    try:
        p = np.asarray(sm_results.pvalues)
        return p
    except Exception:
        return np.full((n_params,), np.nan, dtype=float)


def probe_content(df: pd.DataFrame, *, model_name: str, max_features: int):
    """
    content-only probing (tf-idf)

    returns:
      - classifiers: logistic / mlp / xgboost (mean acc + ci, avg feature weights)
      - statsmodels: coefficients + p-values (logit or mnlogit). if p-values cannot be
        computed due to covariance issues, p_value is set to NaN instead of crashing.
    """
    results = {}

    vectorizer = build_content_vectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["response"]).toarray()
    feature_names = vectorizer.get_feature_names_out()

    y, classes, label_kind = encode_labels_for_statsmodels(df)
    seeds = sorted(df["seed"].unique())
    splits = [(df["seed"] != s, df["seed"] == s) for s in seeds]

    model_defs = {
        "logistic": lambda: LogisticRegression(
            C=1.0, max_iter=1000, solver="liblinear", penalty="l2", random_state=42
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=2000,
            early_stopping=True,
            random_state=42,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        ),
    }

    for name, constructor in model_defs.items():
        accs, weights = [], []
        for train_idx, test_idx in splits:
            clf = constructor()
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], preds))
            weights.append(get_feature_weights(clf, feature_names, name))

        mean_acc, ci = compute_ci(accs)

        avg_weights = (
            pd.concat(weights)
            .groupby("feature", as_index=False)["weight"]
            .mean()
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )

        if name == "xgboost":
            mapping = {f"f{i}": feature_names[i] for i in range(len(feature_names))}
            avg_weights["feature"] = avg_weights["feature"].map(mapping)

        results[name] = {"mean_acc": mean_acc, "ci": ci, "feature_weights": avg_weights}

    # statsmodels (robust to covariance failures)
    try:
        vectorizer_stats = build_content_vectorizer(max_features=max_features)
        X_stats = vectorizer_stats.fit_transform(df["response"]).toarray()
        feature_names_stats = vectorizer_stats.get_feature_names_out()

        X_const = sm.add_constant(X_stats)
        n_classes = len(np.unique(y))

        if n_classes == 2:
            sm_res = sm.Logit(y, X_const).fit(disp=0, maxiter=2000, method="lbfgs")
            params = np.asarray(sm_res.params)
            pvals = safe_get_pvalues(sm_res, n_params=len(params))
            feat_const = ["const"] + list(feature_names_stats)

            stats_df = pd.DataFrame(
                {
                    "feature": feat_const[1:],
                    "class": "M",
                    "coef": params[1:],
                    "p_value": pvals[1:],
                }
            )
        else:
            sm_res = sm.MNLogit(y, X_const).fit(disp=0, maxiter=2000, method="lbfgs")
            params = np.asarray(sm_res.params).flatten()
            pvals = safe_get_pvalues(sm_res, n_params=len(params))
            feat_const = ["const"] + list(feature_names_stats)

            feats_exp, classes_exp = [], []

            class_map = {}
            if label_kind == "race_ethnicity":
                class_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
            elif label_kind == "patron_type":
                class_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

            for feat in feat_const:
                for c in range(n_classes - 1):
                    feats_exp.append(feat)
                    classes_exp.append(str(class_map.get(c, c)))

            stats_df = pd.DataFrame(
                {
                    "feature": feats_exp,
                    "class": classes_exp,
                    "coef": params,
                    "p_value": pvals,
                }
            )

            stats_df = stats_df[stats_df["feature"] != "const"].reset_index(drop=True)

        stats_df = stats_df.dropna(subset=["coef"]).reset_index(drop=True)
        stats_df = stats_df.loc[
            stats_df["coef"].abs().sort_values(ascending=False).index
        ].reset_index(drop=True)

    except Exception as e:
        # do not fail the whole run if statsmodels cannot fit
        stats_df = pd.DataFrame(columns=["feature", "class", "coef", "p_value"])
        results["statsmodels_error"] = f"{type(e).__name__}: {e}"

    results["statsmodels"] = stats_df
    return results


def serialize_for_json(results):
    def convert(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.ndarray, list)):
            return [convert(i) for i in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(results)


def main():
    parser = argparse.ArgumentParser(
        description="run content-only attribute probing (open models only)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["academic", "public"],
        choices=["academic", "public"],
        help="which domains to probe",
    )
    parser.add_argument(
        "--output_path",
        default="probe.json",
        help="where to write the json results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run a single small probe and expose any errors",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="run llama temperature ablation (0.0 vs 0.3), academic only; writes probe_ablation.json",
    )
    args = parser.parse_args()

    characteristics = ["sex", "race_ethnicity", "patron_type"]

    if args.debug:
        domain = "public" if "public" in args.domains else args.domains[0]
        input_dir = DOMAIN_TO_DIR[domain]
        model = "google/gemma-2-9b-it"
        char = "race_ethnicity"

        df = load_data(model, char, input_dir=input_dir)
        results = probe_content(df, model_name=model, max_features=TFIDF_MAX_FEATURES)

        print(
            f"debug domain={domain} input_dir={input_dir} model={model} char={char} max_features={TFIDF_MAX_FEATURES}"
        )
        print("debug statsmodels head:")
        print(results["statsmodels"].head(20).to_string(index=False))
        if "statsmodels_error" in results:
            print("debug statsmodels_error:", results["statsmodels_error"])
        sys.exit(0)

    if args.ablation:
        model = "meta-llama/Llama-3.1-8B-Instruct"
        temps = [0.0, 0.3]
        domain = "academic"
        input_dir = DOMAIN_TO_DIR[domain]

        all_results = {}
        total = len(temps) * len(characteristics)
        progress = tqdm(total=total, desc="running llama temp ablation (content)")

        for temp in temps:
            model_tag = f"{model} [temp={temp}]"
            all_results[model_tag] = {}
            for char in characteristics:
                df = load_data(
                    model,
                    char,
                    input_dir=input_dir,
                    temperature_filter=temp,
                )
                results = probe_content(
                    df,
                    model_name=model,
                    max_features=TFIDF_MAX_FEATURES,
                )
                all_results[model_tag][char] = results
                progress.update(1)

        progress.close()
        with open("probe_ablation.json", "w", encoding="utf-8") as f:
            json.dump(serialize_for_json(all_results), f, ensure_ascii=False, indent=2)

        print("wrote probe_ablation.json")
        return

    # main run: open models only, content only
    all_results = {}
    total = len(args.domains) * len(OPEN_MODEL_NAMES) * len(characteristics)
    progress = tqdm(total=total, desc="running probes (open models, content only)")

    for domain in args.domains:
        input_dir = DOMAIN_TO_DIR[domain]
        all_results[domain] = {}

        for model in OPEN_MODEL_NAMES:
            all_results[domain][model] = {}

            for char in characteristics:
                df = load_data(model, char, input_dir=input_dir)
                results = probe_content(
                    df,
                    model_name=model,
                    max_features=TFIDF_MAX_FEATURES,
                )
                all_results[domain][model][char] = results
                progress.update(1)

    progress.close()

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(serialize_for_json(all_results), f, ensure_ascii=False, indent=2)

    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
