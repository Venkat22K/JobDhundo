import os
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# FILE PATHS
# ============================================================

DATASET_FILE = "data/dataset.csv"
KEYWORD_FILE = "results/keyword_results.csv"
SEMANTIC_FILE = "results/semantic_results.csv"

SUMMARY_FILE = "results/evaluation_summary.csv"
DETAIL_FILE = "results/evaluation_per_cv.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

dataset = pd.read_csv(DATASET_FILE)
keyword = pd.read_csv(KEYWORD_FILE)
semantic = pd.read_csv(SEMANTIC_FILE)

print("Dataset shape:", dataset.shape)
print("Keyword results:", keyword.shape)
print("Semantic results:", semantic.shape)


# ============================================================
# SELECT REQUIRED COLUMNS
# ============================================================

reference = dataset[
    [
        "cv_id",
        "vacancy_id",
        "reference_rank"
    ]
].copy()

keyword = keyword[
    [
        "cv_id",
        "vacancy_id",
        "tfidf_score",
        "keyword_rank"
    ]
].copy()

semantic = semantic[
    [
        "cv_id",
        "vacancy_id",
        "semantic_score",
        "semantic_rank"
    ]
].copy()


# ============================================================
# MERGE RESULTS
# ============================================================

results = reference.merge(
    keyword,
    on=["cv_id", "vacancy_id"],
    how="inner"
)

results = results.merge(
    semantic,
    on=["cv_id", "vacancy_id"],
    how="inner"
)


# ============================================================
# VALIDATE DATA
# ============================================================

print()
print("Merged results shape:", results.shape)

expected_rows = 30 * 5

if len(results) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows, "
        f"but found {len(results)}."
    )

print("Row count check: PASS")


# ============================================================
# CHECK REFERENCE RANKINGS
# ============================================================

print()
print("Checking reference rankings...")

for cv_id in sorted(results["cv_id"].unique()):

    cv_data = results[
        results["cv_id"] == cv_id
    ]

    ranks = cv_data[
        "reference_rank"
    ].tolist()

    if sorted(ranks) != [1, 2, 3, 4, 5]:

        raise ValueError(
            f"Invalid reference ranking for CV {cv_id}: {ranks}"
        )

print("Reference ranking check: PASS")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_method(
    data,
    model_rank_column
):

    per_cv = []

    for cv_id in sorted(
        data["cv_id"].unique()
    ):

        cv_data = data[
            data["cv_id"] == cv_id
        ].sort_values("vacancy_id")

        model_ranks = cv_data[
            model_rank_column
        ].to_numpy()

        reference_ranks = cv_data[
            "reference_rank"
        ].to_numpy()

        # ----------------------------------------------------
        # Spearman correlation
        # ----------------------------------------------------

        spearman = spearmanr(
            model_ranks,
            reference_ranks
        ).statistic

        # ----------------------------------------------------
        # Kendall's tau
        # ----------------------------------------------------

        kendall = kendalltau(
            model_ranks,
            reference_ranks
        ).statistic

        # ----------------------------------------------------
        # Top-1 agreement
        # ----------------------------------------------------

        model_best = cv_data.loc[
            cv_data[model_rank_column] == 1,
            "vacancy_id"
        ].iloc[0]

        reference_best = cv_data.loc[
            cv_data["reference_rank"] == 1,
            "vacancy_id"
        ].iloc[0]

        top1 = int(
            model_best == reference_best
        )

        # ----------------------------------------------------
        # Mean absolute rank difference
        # ----------------------------------------------------

        mean_rank_difference = (
            abs(
                model_ranks -
                reference_ranks
            ).mean()
        )

        per_cv.append({

            "cv_id": cv_id,

            "spearman": spearman,

            "kendall_tau": kendall,

            "top1_agreement": top1,

            "mean_rank_difference":
                mean_rank_difference

        })

    per_cv_df = pd.DataFrame(
        per_cv
    )

    summary = {

        "mean_spearman":
            per_cv_df["spearman"].mean(),

        "mean_kendall_tau":
            per_cv_df["kendall_tau"].mean(),

        "top1_accuracy":
            per_cv_df["top1_agreement"].mean(),

        "mean_rank_difference":
            per_cv_df[
                "mean_rank_difference"
            ].mean(),

        "total_cvs":
            len(per_cv_df)

    }

    return summary, per_cv_df


# ============================================================
# EVALUATE TF-IDF
# ============================================================

print()
print("Evaluating TF-IDF...")

tfidf_summary, tfidf_detail = evaluate_method(
    results,
    "keyword_rank"
)


# ============================================================
# EVALUATE SBERT
# ============================================================

print()
print("Evaluating SBERT...")

sbert_summary, sbert_detail = evaluate_method(
    results,
    "semantic_rank"
)


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame([
    {
        "method": "TF-IDF",
        **tfidf_summary
    },
    {
        "method": "SBERT",
        **sbert_summary
    }
])


# ============================================================
# CREATE DETAIL TABLE
# ============================================================

tfidf_detail["method"] = "TF-IDF"
sbert_detail["method"] = "SBERT"

detail_df = pd.concat(
    [
        tfidf_detail,
        sbert_detail
    ],
    ignore_index=True
)

# Put method first
detail_df = detail_df[
    [
        "method",
        "cv_id",
        "spearman",
        "kendall_tau",
        "top1_agreement",
        "mean_rank_difference"
    ]
]


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

detail_df.to_csv(
    DETAIL_FILE,
    index=False
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print()
print("========================================")
print("EVALUATION COMPLETE")
print("========================================")

print()

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Summary saved to:")
print(SUMMARY_FILE)

print()
print("Per-CV results saved to:")
print(DETAIL_FILE)