import shutil
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


# ═════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════

CSV_PATH = "Data/annotations.csv"

DATASET_DIR = Path("Data/ADNI_AXIAL_NPY_15")

OUTPUT_DIR = Path("Data/ADNI_FINAL_2")

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ═════════════════════════════════════════════════════
# COLUNAS DO CSV
# ═════════════════════════════════════════════════════

SUBJECT_COL = "Subject"
CLASS_COL = "Group"
DATE_COL = "Acq Date"


# ═════════════════════════════════════════════════════
# LOAD CSV
# ═════════════════════════════════════════════════════

df = pd.read_csv(CSV_PATH)

print(f"Registros originais: {len(df)}")


# ═════════════════════════════════════════════════════
# CONVERTER DATA
# ═════════════════════════════════════════════════════

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)


# ═════════════════════════════════════════════════════
# ORDENAR TEMPORALMENTE
# ═════════════════════════════════════════════════════

df = df.sort_values(
    by=[SUBJECT_COL, DATE_COL]
)


# ═════════════════════════════════════════════════════
# PRIMEIRA VISITA DE CADA PACIENTE
# ═════════════════════════════════════════════════════

df_first = (
    df
    .groupby(SUBJECT_COL)
    .first()
    .reset_index()
)

print(f"Primeiras visitas: {len(df_first)}")


# ═════════════════════════════════════════════════════
# PACIENTES EXISTENTES NO DATASET
# ═════════════════════════════════════════════════════

existing_patients = {
    p.name
    for p in DATASET_DIR.iterdir()
    if p.is_dir()
}

df_first = df_first[
    df_first[SUBJECT_COL].isin(existing_patients)
]

print(f"Pacientes encontrados no dataset: {len(df_first)}")


# ═════════════════════════════════════════════════════
# LABELS FINAIS
# ═════════════════════════════════════════════════════

labels_df = pd.DataFrame({
    "subject_id": df_first[SUBJECT_COL],
    "group": df_first[CLASS_COL]
})

print("\nDistribuição das classes:")
print(labels_df["group"].value_counts())


# ═════════════════════════════════════════════════════
# SPLIT ESTRATIFICADO POR PACIENTE
# ═════════════════════════════════════════════════════

train_df, test_df = train_test_split(
    labels_df,
    test_size=TEST_SIZE,
    stratify=labels_df["group"],
    random_state=RANDOM_STATE
)

print(f"\nTrain: {len(train_df)} pacientes")
print(f"Test : {len(test_df)} pacientes")


# ═════════════════════════════════════════════════════
# COPY FUNCTION
# ═════════════════════════════════════════════════════

def copy_patients(split_df, split_name):

    for _, row in split_df.iterrows():

        patient_id = row["subject_id"]
        label = row["group"]

        src = DATASET_DIR / patient_id

        dst = (
            OUTPUT_DIR /
            split_name /
            label /
            patient_id
        )

        dst.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copytree(src, dst, dirs_exist_ok=True)

    print(f"{split_name}: {len(split_df)} pacientes copiados")


# ═════════════════════════════════════════════════════
# EXECUTAR
# ═════════════════════════════════════════════════════

copy_patients(train_df, "train")
copy_patients(test_df, "test")


# ═════════════════════════════════════════════════════
# SALVAR LABELS
# ═════════════════════════════════════════════════════

train_df.to_csv(
    OUTPUT_DIR / "train_labels.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test_labels.csv",
    index=False
)

print("\nDataset final criado com sucesso.")
print("\nTRAIN")
print(train_df["group"].value_counts())

print("\nTEST")
print(test_df["group"].value_counts())