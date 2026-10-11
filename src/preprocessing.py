import re
import string
import pandas as pd


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    """
    Clean resume or job-description text.

    Steps:
    1. Convert to lowercase
    2. Remove email addresses
    3. Remove URLs
    4. Remove special characters
    5. Remove extra whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove email addresses
    text = re.sub(
        r'\S+@\S+',
        ' ',
        text
    )

    # Remove URLs
    text = re.sub(
        r'http\S+|www\S+',
        ' ',
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    # Remove extra whitespace
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# ============================================================
# LOAD DATASET
# ============================================================

def preprocess_dataset(
    input_file="data/dataset.csv",
    output_file="data/processed_dataset.csv"
):

    print("Loading dataset...")

    df = pd.read_csv(input_file)

    print(
        "Original dataset shape:",
        df.shape
    )


    # --------------------------------------------------------
    # Clean resumes
    # --------------------------------------------------------

    print("Cleaning resume text...")

    df["clean_resume"] = df["resume_text"].apply(
        clean_text
    )


    # --------------------------------------------------------
    # Clean job descriptions
    # --------------------------------------------------------

    print("Cleaning job descriptions...")

    df["clean_job_description"] = df[
        "job_description"
    ].apply(
        clean_text
    )


    # --------------------------------------------------------
    # Check empty text
    # --------------------------------------------------------

    empty_resumes = (
        df["clean_resume"].str.strip() == ""
    ).sum()

    empty_jobs = (
        df["clean_job_description"].str.strip() == ""
    ).sum()


    print()
    print(
        "Empty resumes after preprocessing:",
        empty_resumes
    )

    print(
        "Empty job descriptions after preprocessing:",
        empty_jobs
    )


    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )


    print()
    print(
        "Processed dataset saved to:",
        output_file
    )

    print(
        "Processed dataset shape:",
        df.shape
    )

    return df


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    preprocess_dataset()