import pandas as pd

df = pd.read_csv("data/dataset.csv")

missing = df[df["resume_text"].isna()]

print(missing[["cv_id", "vacancy_id"]])
print("Problematic CVs:", missing["cv_id"].unique())