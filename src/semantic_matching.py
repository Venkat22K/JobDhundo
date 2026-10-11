import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/processed_dataset.csv"
OUTPUT_FILE = "results/semantic_results.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# LOAD SBERT MODEL
# ============================================================

print()
print("Loading SBERT model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("SBERT model loaded successfully.")


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

print()
print("Calculating semantic similarity...")


scores = []


for cv_id in sorted(df["cv_id"].unique()):

    # --------------------------------------------------------
    # Get data for this CV
    # --------------------------------------------------------

    cv_data = df[
        df["cv_id"] == cv_id
    ].copy()

    resume_text = cv_data[
        "clean_resume"
    ].iloc[0]

    job_descriptions = cv_data[
        "clean_job_description"
    ].tolist()


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    resume_embedding = model.encode(
        [resume_text]
    )

    job_embeddings = model.encode(
        job_descriptions
    )


    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]


    # --------------------------------------------------------
    # Store scores
    # --------------------------------------------------------

    for i, similarity in enumerate(
        similarities
    ):

        vacancy_id = i + 1

        scores.append({

            "cv_id": cv_id,

            "vacancy_id": vacancy_id,

            "semantic_score": similarity

        })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(scores)


# ============================================================
# CREATE MODEL RANKING
# ============================================================

# Higher semantic similarity = better match

results["semantic_rank"] = (
    results
    .groupby("cv_id")["semantic_score"]
    .rank(
        ascending=False,
        method="min"
    )
)


# Convert rank to integer

results["semantic_rank"] = (
    results["semantic_rank"]
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
print("SEMANTIC MATCHING COMPLETE")
print("========================================")

print(
    "Total resume-job pairs:",
    len(results)
)

print(
    "Output:",
    OUTPUT_FILE
)


# ============================================================
# SHOW EXAMPLE
# ============================================================

print()
print("Example ranking for CV 1:")

example = results[
    results["cv_id"] == 1
].sort_values(
    "semantic_rank"
)


print(
    example[
        [
            "cv_id",
            "vacancy_id",
            "semantic_score",
            "semantic_rank"
        ]
    ].to_string(index=False)
)