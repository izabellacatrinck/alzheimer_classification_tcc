#!/usr/bin/env python3
"""
================================================================================
MEDSIGLIP LINEAR PROBE - BINARY CLASSIFICATION
================================================================================

Objetivo:
    Avaliar a qualidade dos embeddings do MedSigLIP utilizando
    um classificador supervisionado linear (Logistic Regression).

Metodologia:
    - Embeddings extraídos do MedSigLIP
    - Train/Test separados por paciente
    - Apenas 1 exame por paciente
    - MRI fatiada em slices 2D
    - Embedding final do paciente = média dos embeddings dos slices
    - Classificação binária em pares:
        * CN vs AD
        * MCI vs AD
        * MCI vs CN

Por que Linear Probe?
    O linear probe mede o quanto os embeddings já são
    linearmente separáveis sem fine-tuning completo.

Interpretação:
    - Zero-shot ruim + Linear Probe bom:
        => embeddings úteis, problema é alinhamento textual

    - Linear Probe ruim:
        => embeddings não separáveis

================================================================================
"""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from huggingface_hub import login
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# ==============================================================================
# CONFIG
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "google/medsiglip-448"

TRAIN_DIR = Path("Data/ADNI_FINAL/train")
TEST_DIR = Path("Data/ADNI_FINAL/test")

TRAIN_LABELS_PATH = Path("Data/SPLITS/train_labels.csv")
TEST_LABELS_PATH = Path("Data/SPLITS/test_labels.csv")

RESULTS_DIR = Path("tcc_results_linear_probe")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_BATCH_SIZE = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")
print(f"Model : {MODEL_NAME}")


# ==============================================================================
# LOGIN HF
# ==============================================================================

hf_token = os.getenv("MEDSIG_TOKEN")

if hf_token:
    login(hf_token)


# ==============================================================================
# LOAD MODEL
# ==============================================================================

print("\nLoading model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

model.eval()

print("✓ Model loaded")


# ==============================================================================
# TASKS
# ==============================================================================

BINARY_TASKS = [
    {
        "class1": "CN",
        "class2": "AD",
        "name": "CN vs AD",
    },
    {
        "class1": "MCI",
        "class2": "AD",
        "name": "MCI vs AD",
    },
    {
        "class1": "MCI",
        "class2": "CN",
        "name": "MCI vs CN",
    },
]


# ==============================================================================
# HELPERS
# ==============================================================================

def load_patient_slices(patient_dir):

    patient_dir = Path(patient_dir)

    npy_files = sorted(patient_dir.glob("*.npy"))

    if not npy_files:
        raise ValueError(f"No slices found in {patient_dir}")

    images = []

    for file in npy_files:

        try:

            img = np.load(file)

            if np.isnan(img).any():
                continue

            if np.isinf(img).any():
                continue

            if img.max() == img.min():
                continue

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            img = (img * 255).astype(np.uint8)

            pil_img = Image.fromarray(img).convert("RGB")

            images.append(pil_img)

        except Exception:
            continue

    if len(images) == 0:
        raise ValueError(f"No valid slices in {patient_dir}")

    return images


def compute_patient_embedding(images):

    embeddings = []

    for start in range(0, len(images), IMAGE_BATCH_SIZE):

        batch_images = images[start:start + IMAGE_BATCH_SIZE]

        inputs = processor(
            images=batch_images,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():

            outputs = model.get_image_features(**inputs)

            if hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output

            elif isinstance(outputs, torch.Tensor):
                features = outputs

            else:
                features = outputs[0]


            embeddings.append(
                features.detach().cpu().numpy()
            )

    embeddings = np.vstack(embeddings)

    patient_embedding = np.max(embeddings, axis=0)

    return patient_embedding.astype(np.float32)


def load_records(labels_path, split_dir):

    df = pd.read_csv(labels_path)

    records = []

    for _, row in df.iterrows():

        patient_id = row["subject_id"]

        label = row["group"]

        patient_dir = Path(split_dir) / label / patient_id

        if patient_dir.exists():

            records.append({
                "patient_id": patient_id,
                "label": label,
                "patient_dir": patient_dir,
            })

    return records


def attach_embeddings(records, split_name):

    processed = []

    for record in tqdm(records, desc=f"Embedding {split_name}"):

        try:

            images = load_patient_slices(record["patient_dir"])

            embedding = compute_patient_embedding(images)

            processed.append({
                **record,
                "embedding": embedding,
                "n_slices": len(images),
            })

        except Exception:
            continue

    return processed


def compute_metrics(y_true, y_pred, y_score, class1, class2):

    y_true_binary = np.array([
        0 if y == class1 else 1
        for y in y_true
    ])

    metrics = {

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),

        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),

        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[class1, class2]
        ),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[class1, class2],
            zero_division=0
        )
    }

    try:

        metrics["auc"] = float(
            roc_auc_score(
                y_true_binary,
                y_score
            )
        )

    except Exception:

        metrics["auc"] = 0.0

    return metrics


def save_report(
    output_path,
    title,
    metrics,
    class1,
    class2,
    extra_lines=None
):

    with open(output_path, "w", encoding="utf-8") as f:

        f.write(title + "\n")
        f.write("=" * 80 + "\n\n")

        f.write("METRICS\n")
        f.write("-" * 80 + "\n")

        f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Precision Weighted: {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall Weighted:    {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1 Weighted:        {metrics['f1_weighted']:.4f}\n")
        f.write(f"F1 Macro:           {metrics['f1_macro']:.4f}\n")
        f.write(f"AUC:                {metrics['auc']:.4f}\n\n")

        if extra_lines:

            f.write("DIAGNOSTICS\n")
            f.write("-" * 80 + "\n")

            for line in extra_lines:
                f.write(f"{line}\n")

            f.write("\n")

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n")

        f.write(str(metrics["confusion_matrix"]))

        f.write("\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n")

        f.write(metrics["classification_report"])


# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\nLoading records...")

train_records = load_records(
    TRAIN_LABELS_PATH,
    TRAIN_DIR
)

test_records = load_records(
    TEST_LABELS_PATH,
    TEST_DIR
)

print(f"Train patients: {len(train_records)}")
print(f"Test patients : {len(test_records)}")


# ==============================================================================
# COMPUTE EMBEDDINGS
# ==============================================================================

train_records = attach_embeddings(
    train_records,
    "train"
)

test_records = attach_embeddings(
    test_records,
    "test"
)

print(f"\nEmbedded train: {len(train_records)}")
print(f"Embedded test : {len(test_records)}")


# ==============================================================================
# MAIN LOOP
# ==============================================================================

summary_rows = []

all_results = {}

for task in BINARY_TASKS:

    class1 = task["class1"]
    class2 = task["class2"]

    task_name = task["name"]

    print("\n" + "=" * 80)
    print(task_name)
    print("=" * 80)

    task_dir = RESULTS_DIR / f"{class1}_vs_{class2}"

    task_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # FILTER TASK
    # ==========================================================================

    train_task = [
        r for r in train_records
        if r["label"] in [class1, class2]
    ]

    test_task = [
        r for r in test_records
        if r["label"] in [class1, class2]
    ]

    # ==========================================================================
    # BUILD MATRICES
    # ==========================================================================

    X_train = np.vstack([
        r["embedding"]
        for r in train_task
    ])

    y_train = np.array([
        0 if r["label"] == class1 else 1
        for r in train_task
    ])

    X_test = np.vstack([
        r["embedding"]
        for r in test_task
    ])

    y_test = np.array([
        0 if r["label"] == class1 else 1
        for r in test_task
    ])

    y_test_labels = [
        r["label"]
        for r in test_task
    ]

    # ==========================================================================
    # STANDARDIZATION
    # ==========================================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)
    pca = PCA(
    n_components=128,
    random_state=SEED
)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # ==========================================================================
    # LOGISTIC REGRESSION
    # ==========================================================================

    clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=SEED,
)

    clf.fit(X_train, y_train)

    # ==========================================================================
    # PREDICTIONS
    # ==========================================================================

    y_pred_binary = clf.predict(X_test)

    y_prob = clf.predict_proba(X_test)[:, 1]

    y_pred_labels = [
        class1 if y == 0 else class2
        for y in y_pred_binary
    ]

    # ==========================================================================
    # METRICS
    # ==========================================================================

    metrics = compute_metrics(
        y_true=y_test_labels,
        y_pred=y_pred_labels,
        y_score=y_prob,
        class1=class1,
        class2=class2,
    )

    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Weighted:       {metrics['f1_weighted']:.4f}")
    print(f"F1 Macro:          {metrics['f1_macro']:.4f}")
    print(f"AUC:               {metrics['auc']:.4f}")

    # ==========================================================================
    # SAVE CSV
    # ==========================================================================

    predictions_df = pd.DataFrame({

        "subject_id": [
            r["patient_id"]
            for r in test_task
        ],

        "true_label": y_test_labels,

        "pred_label": y_pred_labels,

        "prob_class2": y_prob,
    })

    predictions_df.to_csv(
        task_dir / "predictions.csv",
        index=False
    )

    # ==========================================================================
    # SAVE REPORT
    # ==========================================================================

    save_report(
        output_path=task_dir / "report.txt",
        title=f"LINEAR PROBE - {task_name}",
        metrics=metrics,
        class1=class1,
        class2=class2,
        extra_lines=[
            f"Model: {MODEL_NAME}",
            "Classifier: Logistic Regression",
            "Embeddings: Mean patient embedding",
            "Protocol: Train/Test split by patient",
            "MRI representation: 2D slices",
            "One exam per patient",
        ]
    )

    # ==========================================================================
    # CONFUSION MATRIX
    # ==========================================================================

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class1, class2],
        yticklabels=[class1, class2],
    )

    plt.title(
        f"{task_name}\nAcc={metrics['accuracy']:.3f}"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()

    plt.savefig(
        task_dir / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================================================
    # STORE SUMMARY
    # ==========================================================================

    summary_rows.append({

        "Task": task_name,

        "Accuracy": metrics["accuracy"],

        "Balanced_Accuracy": metrics["balanced_accuracy"],

        "F1_Weighted": metrics["f1_weighted"],

        "F1_Macro": metrics["f1_macro"],

        "AUC": metrics["auc"],
    })

    all_results[f"{class1}_vs_{class2}"] = metrics


# ==============================================================================
# SAVE SUMMARY
# ==============================================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    RESULTS_DIR / "summary.csv",
    index=False
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(summary_df.to_string(index=False))

# ==============================================================================
# SAVE JSON
# ==============================================================================

with open(
    RESULTS_DIR / "all_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        default=str
    )

print(f"\n✓ Results saved to: {RESULTS_DIR}")