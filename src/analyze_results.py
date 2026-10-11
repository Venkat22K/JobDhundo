import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# FILE PATHS
# ============================================================

DATASET_FILE = "data/dataset.csv"
KEYWORD_FILE = "results/keyword_results.csv"
SEMANTIC_FILE = "results/semantic_results.csv"

OUTPUT_DIR = "results"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading results...")

dataset = pd.read_csv(DATASET_FILE)
keyword = pd.read_csv(KEYWORD_FILE)
semantic = pd.read_csv(SEMANTIC_FILE)

print("Dataset:", dataset.shape)
print("Keyword:", keyword.shape)
print("Semantic:", semantic.shape)


# ============================================================
# MERGE EVERYTHING
# ============================================================

data = dataset[
    [
        "cv_id",
        "vacancy_id",
        "reference_rank"
    ]
].merge(
    keyword[
        [
            "cv_id",
            "vacancy_id",
            "keyword_rank",
            "tfidf_score"
        ]
    ],
    on=["cv_id", "vacancy_id"]
).merge(
    semantic[
        [
            "cv_id",
            "vacancy_id",
            "semantic_rank",
            "semantic_score"
        ]
    ],
    on=["cv_id", "vacancy_id"]
)


# ============================================================
# CALCULATE PER-CV ANALYSIS
# ============================================================

analysis = []


for cv_id in sorted(data["cv_id"].unique()):

    cv = data[
        data["cv_id"] == cv_id
    ].sort_values("vacancy_id")

    reference = cv["reference_rank"].tolist()
    tfidf = cv["keyword_rank"].tolist()
    sbert = cv["semantic_rank"].tolist()

    # --------------------------------------------------------
    # Rank differences
    # --------------------------------------------------------

    tfidf_error = abs(
        cv["keyword_rank"] -
        cv["reference_rank"]
    ).mean()

    sbert_error = abs(
        cv["semantic_rank"] -
        cv["reference_rank"]
    ).mean()

    # --------------------------------------------------------
    # Difference between TF-IDF and SBERT
    # --------------------------------------------------------

    model_rank_difference = abs(
        cv["keyword_rank"] -
        cv["semantic_rank"]
    ).mean()

    # --------------------------------------------------------
    # Spearman correlation between the two models
    # --------------------------------------------------------

    model_correlation = spearmanr(
        cv["keyword_rank"],
        cv["semantic_rank"]
    ).statistic

    # --------------------------------------------------------
    # Top-1 jobs
    # --------------------------------------------------------

    reference_best = cv.loc[
        cv["reference_rank"] == 1,
        "vacancy_id"
    ].iloc[0]

    tfidf_best = cv.loc[
        cv["keyword_rank"] == 1,
        "vacancy_id"
    ].iloc[0]

    sbert_best = cv.loc[
        cv["semantic_rank"] == 1,
        "vacancy_id"
    ].iloc[0]

    analysis.append({

        "cv_id": cv_id,

        "reference_ranking": str(reference),

        "tfidf_ranking": str(tfidf),

        "sbert_ranking": str(sbert),

        "tfidf_error": tfidf_error,

        "sbert_error": sbert_error,

        "model_rank_difference":
            model_rank_difference,

        "model_spearman":
            model_correlation,

        "reference_best":
            reference_best,

        "tfidf_best":
            tfidf_best,

        "sbert_best":
            sbert_best,

        "tfidf_top1_correct":
            int(tfidf_best == reference_best),

        "sbert_top1_correct":
            int(sbert_best == reference_best)

    })


analysis_df = pd.DataFrame(analysis)


# ============================================================
# SORT BY ACTUAL MODEL DISAGREEMENT
# ============================================================

model_disagreement = analysis_df.sort_values(
    "model_rank_difference",
    ascending=False
)


# ============================================================
# SAVE ANALYSIS
# ============================================================

model_disagreement.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_disagreement.csv"
    ),
    index=False
)


# ============================================================
# DISPLAY MOST DIFFERENT CASES
# ============================================================

print()
print("========================================")
print("TF-IDF vs SBERT: MOST DIFFERENT CASES")
print("========================================")

print()

print(
    model_disagreement[
        [
            "cv_id",
            "reference_ranking",
            "tfidf_ranking",
            "sbert_ranking",
            "model_rank_difference",
            "model_spearman",
            "reference_best",
            "tfidf_best",
            "sbert_best"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# DISPLAY CASES WHERE ONE METHOD DOES MUCH BETTER
# ============================================================

analysis_df["tfidf_vs_sbert_error_gap"] = (
    analysis_df["tfidf_error"] -
    analysis_df["sbert_error"]
)

analysis_df["absolute_error_gap"] = (
    analysis_df["tfidf_vs_sbert_error_gap"].abs()
)

error_gap = analysis_df.sort_values(
    "absolute_error_gap",
    ascending=False
)


print()
print("========================================")
print("CASES WITH LARGEST PERFORMANCE GAP")
print("========================================")

print()

print(
    error_gap[
        [
            "cv_id",
            "reference_ranking",
            "tfidf_ranking",
            "sbert_ranking",
            "tfidf_error",
            "sbert_error",
            "tfidf_vs_sbert_error_gap"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# GRAPH 1: MODEL RANK DIFFERENCE
# ============================================================

plot_data = model_disagreement.head(15)

plt.figure(figsize=(10, 6))

plt.bar(
    plot_data["cv_id"].astype(str),
    plot_data["model_rank_difference"]
)

plt.title(
    "TF-IDF vs SBERT Ranking Difference"
)

plt.xlabel("CV ID")
plt.ylabel("Mean Absolute Rank Difference")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "tfidf_vs_sbert_difference.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2: ERROR COMPARISON
# ============================================================

avg_errors = pd.DataFrame({
    "Method": ["TF-IDF", "SBERT"],
    "Mean Rank Error": [
        analysis_df["tfidf_error"].mean(),
        analysis_df["sbert_error"].mean()
    ]
})

plt.figure(figsize=(8, 5))

plt.bar(
    avg_errors["Method"],
    avg_errors["Mean Rank Error"]
)

plt.title(
    "Average Rank Error"
)

plt.xlabel("Method")
plt.ylabel("Mean Absolute Rank Difference")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "average_rank_error.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print()
print("Generated:")

print(
    "results/model_disagreement.csv"
)

print(
    "results/tfidf_vs_sbert_difference.png"
)

print(
    "results/average_rank_error.png"
)