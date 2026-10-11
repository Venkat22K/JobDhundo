import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/processed_dataset.csv"
OUTPUT_FILE = "results/keyword_results.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# TF-IDF + COSINE SIMILARITY
# ============================================================

print()
print("Calculating TF-IDF similarity...")


scores = []


for cv_id in sorted(df["cv_id"].unique()):

    # Get all rows belonging to this CV
    cv_data = df[df["cv_id"] == cv_id].copy()

    resume_text = cv_data["clean_resume"].iloc[0]

    job_descriptions = cv_data[
        "clean_job_description"
    ].tolist()

    # --------------------------------------------------------
    # Create TF-IDF representation
    # --------------------------------------------------------

    documents = [
        resume_text
    ] + job_descriptions

    vectorizer = TfidfVectorizer(
        lowercase=True
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    # First vector = resume
    resume_vector = tfidf_matrix[0]

    # Remaining vectors = jobs
    job_vectors = tfidf_matrix[1:]

    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]

    # --------------------------------------------------------
    # Store scores
    # --------------------------------------------------------

    for i, similarity in enumerate(similarities):

        vacancy_id = i + 1

        scores.append({

            "cv_id": cv_id,

            "vacancy_id": vacancy_id,

            "tfidf_score": similarity

        })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(scores)


# ============================================================
# CREATE MODEL RANKING
# ============================================================

# Higher similarity = better match
results["keyword_rank"] = (
    results
    .groupby("cv_id")["tfidf_score"]
    .rank(
        ascending=False,
        method="min"
    )
)


# Convert rank to integer
results["keyword_rank"] = (
    results["keyword_rank"]
    .astype(int)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("========================================")
print("KEYWORD MATCHING COMPLETE")
print("========================================")

print(
    "Total resume-job pairs:",
    len(results)
)

print(
    "Output:",
    OUTPUT_FILE
)


# Show first CV as an example

print()
print("Example ranking for CV 1:")

example = results[
    results["cv_id"] == 1
].sort_values(
    "keyword_rank"
)

print(
    example[
        [
            "cv_id",
            "vacancy_id",
            "tfidf_score",
            "keyword_rank"
        ]
    ].to_string(index=False)
)