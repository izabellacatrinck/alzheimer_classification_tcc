from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split


# =====================================================
# CONFIG
# =====================================================

CSV_PATH = "Data/annotations.csv"

DATASET_DIR = Path("Data/ADNI_REDUZIDO")

OUTPUT_DIR = Path("Data/SPLITS")

TEST_SIZE = 0.20
RANDOM_STATE = 42

SUBJECT_COL = "Subject"
CLASS_COL = "Group"
DATE_COL = "Acq Date"


# =====================================================
# LOAD CSV
# =====================================================

df = pd.read_csv(CSV_PATH)

print(f"Registros originais: {len(df)}")


# =====================================================
# DATE CONVERSION
# =====================================================

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)


# =====================================================
# SORT TEMPORALLY
# =====================================================

df = df.sort_values(
    by=[SUBJECT_COL, DATE_COL]
)


# =====================================================
# FIRST VISIT ONLY
# =====================================================

df_first = (
    df
    .groupby(SUBJECT_COL)
    .first()
    .reset_index()
)

print(f"Primeiras visitas: {len(df_first)}")


# =====================================================
# VALID PATIENTS
# Must contain:
#   - raw MRI
#   - preprocessed MRI
# =====================================================

valid_patients = []

for patient_dir in DATASET_DIR.iterdir():

    if not patient_dir.is_dir():
        continue

    subject_id = patient_dir.name

    # -----------------------------------------
    # PREPROCESSED MRI
    # -----------------------------------------

    preprocessed_files = list(
        patient_dir.glob("*_resampled.nii.gz")
    )

    # -----------------------------------------
    # RAW MRI
    # -----------------------------------------

    raw_files = [
        f for f in patient_dir.iterdir()
        if (
            f.is_file()
            and (
                f.suffix == ".nii"
                or f.name.endswith(".nii.gz")
            )
            and "_resampled" not in f.name
        )
    ]

    # -----------------------------------------
    # VALIDATION
    # -----------------------------------------

    if len(preprocessed_files) == 0:
        continue

    if len(raw_files) == 0:
        continue

    valid_patients.append(subject_id)


valid_patients = set(valid_patients)

print(f"Pacientes válidos: {len(valid_patients)}")


# =====================================================
# FILTER VALID SUBJECTS
# =====================================================

df_first = df_first[
    df_first[SUBJECT_COL].isin(valid_patients)
]


# =====================================================
# LABELS
# =====================================================

labels_df = pd.DataFrame({
    "subject_id": df_first[SUBJECT_COL],
    "group": df_first[CLASS_COL]
})

print("\nDistribuição das classes:")
print(labels_df["group"].value_counts())


# =====================================================
# STRATIFIED SPLIT
# =====================================================

train_df, test_df = train_test_split(
    labels_df,
    test_size=TEST_SIZE,
    stratify=labels_df["group"],
    random_state=RANDOM_STATE
)

print(f"\nTrain: {len(train_df)}")
print(f"Test : {len(test_df)}")


# =====================================================
# SAVE SPLITS
# =====================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train_df.to_csv(
    OUTPUT_DIR / "train_labels.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test_labels.csv",
    index=False
)


# =====================================================
# OPTIONAL: SAVE VALID SUBJECTS
# =====================================================

pd.DataFrame({
    "subject_id": sorted(valid_patients)
}).to_csv(
    OUTPUT_DIR / "valid_subjects.csv",
    index=False
)


# =====================================================
# FINAL REPORT
# =====================================================

print("\nSplits salvos com sucesso.")

print("\nTRAIN")
print(train_df["group"].value_counts())

print("\nTEST")
print(test_df["group"].value_counts())