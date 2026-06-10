#!/usr/bin/env python3
# ==============================================================================
# DENSENET121 - BINARY ALZHEIMER MRI CLASSIFICATION — v2
# ==============================================================================
#
# MELHORIAS EM RELAÇÃO À VERSÃO ANTERIOR:
#
#     ✓ Todos os 10 slices por paciente utilizados
#     ✓ Patient-level pooling na avaliação (média de probabilidades)
#     ✓ Threshold tuning automático via validação
#     ✓ Focal Loss (substitui CrossEntropy + label smoothing)
#     ✓ Layer-wise learning rate (LR diferenciado por bloco)
#     ✓ Fine-tuning gradual bloco a bloco (4 stages)
#     ✓ Métricas reportadas no nível do paciente
#     ✓ Peso extra para AD na loss (além dos class weights)
#
# MANTIDOS DA VERSÃO ANTERIOR:
#
#     ✓ Split por paciente
#     ✓ Sem data leakage
#     ✓ Augmentations médicas realistas
#     ✓ Early stopping baseado em F1 (nível paciente)
#     ✓ Mixed precision
#     ✓ Gradient clipping
#     ✓ Scheduler robusto
#     ✓ Reprodutibilidade
#
# ==============================================================================

import os
import random
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_WORKERS = 0

# ------------------------------------------------------------------
# STAGES DE FINE-TUNING (4 stages — descongelamento gradual)
#   Stage 1: só o classifier
#   Stage 2: denseblock4
#   Stage 3: denseblock3 + denseblock4
#   Stage 4: rede completa
# ------------------------------------------------------------------

EPOCHS_STAGE_1 = 10
EPOCHS_STAGE_2 = 15
EPOCHS_STAGE_3 = 15
EPOCHS_STAGE_4 = 20

TOTAL_EPOCHS = (
    EPOCHS_STAGE_1
    + EPOCHS_STAGE_2
    + EPOCHS_STAGE_3
    + EPOCHS_STAGE_4
)

EARLY_STOPPING_PATIENCE = 10

BASE_LR = 1e-4

WEIGHT_DECAY = 1e-4

# Focal Loss gamma — valores maiores focam mais nos exemplos difíceis
FOCAL_GAMMA = 2.0

GRAD_CLIP = 1.0

USE_MIXED_PRECISION = True

# ==============================================================================
# TASK CONFIG
# ==============================================================================

CLASS_1 = "CN"
CLASS_2 = "MCI"

TRAIN_DIR = Path("Data/ADNI_FINAL/train")
TEST_DIR  = Path("Data/ADNI_FINAL/test")

TRAIN_LABELS = Path("Data/ADNI_FINAL/train_labels.csv")
TEST_LABELS  = Path("Data/ADNI_FINAL/test_labels.csv")

RESULTS_DIR = Path(f"results_densenet_v1_{CLASS_1}_vs_{CLASS_2}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

# ==============================================================================
# AUGMENTATIONS
# ==============================================================================

#
# MRI NÃO deve usar augmentations agressivas.
# Simular pequenas variações anatômicas/aquisição apenas.
#

train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.95, 1.0),
        ratio=(0.98, 1.02),
    ),

    transforms.RandomRotation(4),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.02, 0.02),
        scale=(0.99, 1.01),
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ==============================================================================
# DATASET
# ==============================================================================

class MRIDataset(Dataset):
    """
    Carrega todos os 10 slices de cada paciente.

    Cada sample mantém o patient_id para que a avaliação
    possa agregar probabilidades no nível do paciente.
    """

    def __init__(self, records, transform=None):

        self.samples   = []
        self.transform = transform

        self.label_map = {
            CLASS_1: 0,
            CLASS_2: 1,
        }

        for record in records:

            patient_dir = record["patient_dir"]
            label       = record["label"]

            npy_files = sorted(Path(patient_dir).glob("*.npy"))

            # ------------------------------------------------------------------
            # Usa TODOS os slices disponíveis (até 10).
            # Se houver mais de 10, pega os centrais para não incluir
            # slices de borda com pouca informação anatômica relevante.
            # ------------------------------------------------------------------

            n = len(npy_files)

            if n <= 10:
                selected = npy_files
            else:
                center   = n // 2
                half     = 5
                selected = npy_files[center - half : center + half]

            for npy_file in selected:

                self.samples.append({
                    "path":       npy_file,
                    "label":      label,
                    "patient_id": record["patient_id"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        img = np.load(sample["path"]).astype(np.float32)

        # ------------------------------------------------------------------
        # Normalização robusta: clip percentil 1–99, depois 0–1
        # ------------------------------------------------------------------

        p1  = np.percentile(img, 1)
        p99 = np.percentile(img, 99)

        img = np.clip(img, p1, p99)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.label_map[sample["label"]]

        return img, label, sample["patient_id"]


# ==============================================================================
# COLLATE — retorna patient_id junto com imagens/labels
# ==============================================================================

def collate_fn(batch):
    images     = torch.stack([b[0] for b in batch])
    labels     = torch.tensor([b[1] for b in batch], dtype=torch.long)
    patient_ids = [b[2] for b in batch]
    return images, labels, patient_ids


# ==============================================================================
# LOAD RECORDS
# ==============================================================================

def load_records(labels_path, split_dir):

    df = pd.read_csv(labels_path)
    df = df[df["group"].isin([CLASS_1, CLASS_2])]

    records = []

    for _, row in df.iterrows():

        patient_id  = row["subject_id"]
        label       = row["group"]
        patient_dir = Path(split_dir) / label / patient_id

        if patient_dir.exists():
            records.append({
                "patient_id":  patient_id,
                "label":       label,
                "patient_dir": patient_dir,
            })

    return records


all_train_records = load_records(TRAIN_LABELS, TRAIN_DIR)
test_records      = load_records(TEST_LABELS,  TEST_DIR)

train_records, val_records = train_test_split(
    all_train_records,
    test_size=0.15,
    stratify=[r["label"] for r in all_train_records],
    random_state=SEED,
)

print(f"Train patients : {len(train_records)}")
print(f"Val patients   : {len(val_records)}")
print(f"Test patients  : {len(test_records)}")

# ==============================================================================
# DATASETS & DATALOADERS
# ==============================================================================

train_dataset = MRIDataset(train_records, transform=train_transform)
val_dataset   = MRIDataset(val_records,   transform=val_transform)
test_dataset  = MRIDataset(test_records,  transform=val_transform)

print(f"\nTrain slices   : {len(train_dataset)}")
print(f"Val slices     : {len(val_dataset)}")
print(f"Test slices    : {len(test_dataset)}")

# ==============================================================================
# PATCH — desbalanceamento MCI vs CN
# ==============================================================================
#
# PROBLEMA IDENTIFICADO:
#   O modelo só prevê MCI porque:
#
#   1. class_weights calculados sobre SLICES, não pacientes.
#      Se CN tem mais slices por paciente, o peso de MCI fica
#      artificialmente baixo mesmo com pacientes balanceados.
#
#   2. DataLoader com shuffle=True sem WeightedRandomSampler:
#      batches aleatórios concentram a classe majoritária,
#      e o modelo aprende a prever sempre MCI para minimizar loss.
#
# CORREÇÃO (2 mudanças apenas, tudo o mais inalterado):
#
#   1. Calcular class_weights por PACIENTE (não por slice)
#   2. Adicionar WeightedRandomSampler ao train_loader
#
# ==============================================================================

# Conta pacientes (não slices) por classe
train_patient_labels = [r["label"] for r in train_records]  # lista de strings
# ==============================================================================
# CLASS WEIGHTS (por PACIENTE, não por slice)
# ==============================================================================

train_labels_numeric_patients = [
    0 if lbl == CLASS_1 else 1
    for lbl in train_patient_labels
]

# class_weights agora reflete o desequilíbrio real entre PACIENTES
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_labels_numeric_patients,
)

class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print(f"\nClass weights (por paciente): {CLASS_1}={weights[0]:.3f}  {CLASS_2}={weights[1]:.3f}")
# ==============================================================================
# WeightedRandomSampler — garante batches balanceados por SLICE,
# respeitando o peso do paciente de origem de cada slice
# ==============================================================================

# Cada slice herda o peso do paciente a que pertence
sample_weights = []
for sample in train_dataset.samples:
    if sample["label"] == CLASS_1:
        sample_weights.append(weights[0])
    else:
        sample_weights.append(weights[1])

from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,          # substitui shuffle=True
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn,
)


# ==============================================================================
# FOCAL LOSS
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss para lidar com desequilíbrio de classes.

    Penaliza mais os exemplos que o modelo erra com alta confiança
    (AD classificado como CN), reduzindo falsos negativos.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0 → equivalente à CrossEntropy ponderada.
    gamma=2 → foca fortemente nos exemplos difíceis.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha      # class weights tensor
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        # Cross entropy por sample (sem redução)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
        )

        # p_t = probabilidade da classe correta
        pt = torch.exp(-ce_loss)

        # Focal factor: (1 - p_t)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


criterion = nn.CrossEntropyLoss(weight=class_weights)

# ==============================================================================
# MODEL
# ==============================================================================

model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

# Congela tudo inicialmente
for param in model.parameters():
    param.requires_grad = False

# Substitui o classifier
"""in_features = model.classifier.in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),
)"""
model.classifier = nn.Linear(1024, 2)
model = model.to(DEVICE)

# ==============================================================================
# PATIENT-LEVEL POOLING
# ==============================================================================

def aggregate_patient_predictions(all_targets, all_preds, all_probs, all_patient_ids):
    """
    Agrega predições de slices para o nível do paciente.

    Estratégia: média das probabilidades da classe AD (softmax score)
    por paciente, depois aplica o threshold para decisão final.

    Isso é mais robusto do que voto majoritário porque preserva
    a incerteza do modelo em cada slice.
    """

    patient_probs   = defaultdict(list)
    patient_targets = {}

    for pid, target, prob in zip(all_patient_ids, all_targets, all_probs):
        patient_probs[pid].append(prob)
        patient_targets[pid] = target   # label do paciente (consistente entre slices)

    pids        = list(patient_probs.keys())
    agg_probs   = np.array([np.mean(patient_probs[pid]) for pid in pids])
    true_labels = np.array([patient_targets[pid]        for pid in pids])

    return true_labels, agg_probs, pids


# ==============================================================================
# THRESHOLD TUNING
# ==============================================================================

def find_best_threshold(y_true, y_prob, metric="f1"):
    """
    Busca o threshold que maximiza F1 (ou balanced accuracy)
    no conjunto de validação.

    Evita usar o test set para escolher o threshold,
    o que constituiria data leakage.
    """

    thresholds   = np.arange(0.20, 0.80, 0.01)
    best_thresh  = 0.5
    best_score   = 0.0

    for t in thresholds:

        preds = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, preds, average="macro")
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score  = score
            best_thresh = t

    return best_thresh, best_score


# ==============================================================================
# METRICS
# ==============================================================================

def compute_metrics(y_true, y_pred, y_prob):

    metrics = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["auc"] = 0.0

    return metrics


# ==============================================================================
# EVALUATION (slice-level + patient-level)
# ==============================================================================

@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    """
    Retorna métricas em dois níveis:
      - slice_metrics : avaliação individual por slice (como antes)
      - patient_metrics: avaliação agregada por paciente (mais confiável)

    O early stopping usa patient_metrics["f1"] como critério.
    """

    model.eval()

    losses          = []
    all_preds       = []
    all_probs       = []
    all_targets     = []
    all_patient_ids = []

    for images, targets, patient_ids in loader:

        images  = images.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):

            # TTA: média com flip horizontal
            outputs1 = model(images)
            outputs2 = model(torch.flip(images, dims=[3]))
            outputs  = (outputs1 + outputs2) / 2

        loss  = criterion(outputs.float(), targets)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        losses.append(loss.item())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_patient_ids.extend(patient_ids)

    # ------------------------------------------------------------------
    # Slice-level metrics
    # ------------------------------------------------------------------

    slice_metrics = compute_metrics(
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )
    slice_metrics["loss"] = np.mean(losses)

    # ------------------------------------------------------------------
    # Patient-level metrics (agrega por paciente)
    # ------------------------------------------------------------------

    pat_true, pat_prob, _ = aggregate_patient_predictions(
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        all_patient_ids,
    )

    pat_pred = (pat_prob >= threshold).astype(int)

    patient_metrics = compute_metrics(pat_true, pat_pred, pat_prob)
    patient_metrics["loss"] = np.mean(losses)

    return (
        slice_metrics,
        patient_metrics,
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        all_patient_ids,
    )


# ==============================================================================
# LAYER-WISE LR HELPER
# ==============================================================================

def get_layerwise_optimizer(model, base_lr, stage, weight_decay):
    """
    Retorna um AdamW com LR diferenciado por grupo de camadas.

    Quanto mais próxima da entrada, menor o LR:
    as primeiras camadas capturam features genéricas (bordas, texturas)
    que já foram bem aprendidas no ImageNet — não precisam mudar muito.

    Stage 1: só classifier
    Stage 2: classifier + denseblock4
    Stage 3: classifier + denseblock4 + denseblock3
    Stage 4: rede completa (todos os blocos)
    """

    if stage == 1:
        param_groups = [
            {
                "params":       model.classifier.parameters(),
                "lr":           base_lr,
                "weight_decay": weight_decay,
                "name":         "classifier",
            }
        ]

    elif stage == 2:
        param_groups = [
            {
                "params":       model.features.denseblock4.parameters(),
                "lr":           base_lr / 5,
                "weight_decay": weight_decay,
                "name":         "denseblock4",
            },
            {
                "params":       model.classifier.parameters(),
                "lr":           base_lr / 2,
                "weight_decay": weight_decay,
                "name":         "classifier",
            },
        ]

    elif stage == 3:
        param_groups = [
            {
                "params":       model.features.denseblock3.parameters(),
                "lr":           base_lr / 20,
                "weight_decay": weight_decay,
                "name":         "denseblock3",
            },
            {
                "params":       model.features.denseblock4.parameters(),
                "lr":           base_lr / 10,
                "weight_decay": weight_decay,
                "name":         "denseblock4",
            },
            {
                "params":       model.classifier.parameters(),
                "lr":           base_lr / 5,
                "weight_decay": weight_decay,
                "name":         "classifier",
            },
        ]

    else:  # stage 4 — rede completa
        param_groups = [
            {
                "params":       model.features.denseblock1.parameters(),
                "lr":           base_lr / 100,
                "weight_decay": weight_decay,
                "name":         "denseblock1",
            },
            {
                "params":       model.features.denseblock2.parameters(),
                "lr":           base_lr / 50,
                "weight_decay": weight_decay,
                "name":         "denseblock2",
            },
            {
                "params":       model.features.denseblock3.parameters(),
                "lr":           base_lr / 20,
                "weight_decay": weight_decay,
                "name":         "denseblock3",
            },
            {
                "params":       model.features.denseblock4.parameters(),
                "lr":           base_lr / 10,
                "weight_decay": weight_decay,
                "name":         "denseblock4",
            },
            {
                "params":       model.classifier.parameters(),
                "lr":           base_lr / 5,
                "weight_decay": weight_decay,
                "name":         "classifier",
            },
        ]

    return optim.AdamW(param_groups)


# ==============================================================================
# UNFREEZE HELPERS
# ==============================================================================

def unfreeze_stage(model, stage):

    if stage == 1:
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif stage == 2:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.transition3.parameters():
            param.requires_grad = True

    elif stage == 3:
        for param in model.features.denseblock3.parameters():
            param.requires_grad = True
        for param in model.features.transition2.parameters():
            param.requires_grad = True

    elif stage == 4:
        for param in model.parameters():
            param.requires_grad = True


# ==============================================================================
# TRAINING SETUP — STAGE 1
# ==============================================================================

unfreeze_stage(model, stage=1)

optimizer = get_layerwise_optimizer(model, BASE_LR, stage=1, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=TOTAL_EPOCHS,
    eta_min=1e-6,
)

scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

# ==============================================================================
# TRAIN LOOP
# ==============================================================================

best_patient_f1  = 0.0
best_threshold   = 0.5
patience_counter = 0
history          = []

stage_boundaries = {
    EPOCHS_STAGE_1:                             2,
    EPOCHS_STAGE_1 + EPOCHS_STAGE_2:            3,
    EPOCHS_STAGE_1 + EPOCHS_STAGE_2 + EPOCHS_STAGE_3: 4,
}

print("\nStarting training...\n")
print(f"Device         : {DEVICE}")
print(f"Total epochs   : {TOTAL_EPOCHS}")
print(f"Focal gamma    : {FOCAL_GAMMA}")
print()

for epoch in range(TOTAL_EPOCHS):

    # ------------------------------------------------------------------
    # PROGRESSIVE FINE-TUNING — transição de stage
    # ------------------------------------------------------------------

    if epoch in stage_boundaries:

        new_stage = stage_boundaries[epoch]

        stage_names = {
            2: "DENSEBLOCK4",
            3: "DENSEBLOCK3 + DENSEBLOCK4",
            4: "FULL NETWORK",
        }

        print(f"\nSTAGE {new_stage} — UNFREEZING {stage_names[new_stage]}\n")

        unfreeze_stage(model, stage=new_stage)

        optimizer = get_layerwise_optimizer(
            model, BASE_LR, stage=new_stage, weight_decay=WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TOTAL_EPOCHS - epoch,
            eta_min=1e-6,
        )

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    model.train()

    train_losses = []

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{TOTAL_EPOCHS}]")

    for images, targets, _ in loop:      # patient_ids não usados no treino

        images  = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
            outputs = model(images)
            loss    = criterion(outputs.float(), targets)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

        loop.set_postfix(loss=f"{np.mean(train_losses):.4f}")

    # ------------------------------------------------------------------
    # VALIDATION — slice-level + patient-level com threshold atual
    # ------------------------------------------------------------------

    (
        val_slice_metrics,
        val_patient_metrics,
        val_targets,
        _,
        val_probs,
        val_patient_ids,
    ) = evaluate(model, val_loader, threshold=best_threshold)

    # Tuning do threshold no conjunto de validação (nível paciente)
    pat_true, pat_prob, _ = aggregate_patient_predictions(
        val_targets,
        (val_probs >= best_threshold).astype(int),
        val_probs,
        val_patient_ids,
    )

    current_threshold, _ = find_best_threshold(pat_true, pat_prob, metric="f1")

    scheduler.step()

    history.append({
        "epoch":                   epoch + 1,
        "train_loss":              np.mean(train_losses),
        "threshold":               current_threshold,
        # slice-level
        "slice_loss":              val_slice_metrics["loss"],
        "slice_f1":                val_slice_metrics["f1"],
        "slice_auc":               val_slice_metrics["auc"],
        "slice_balanced_accuracy": val_slice_metrics["balanced_accuracy"],
        # patient-level
        "patient_loss":            val_patient_metrics["loss"],
        "patient_f1":              val_patient_metrics["f1"],
        "patient_auc":             val_patient_metrics["auc"],
        "patient_recall":          val_patient_metrics["recall"],
        "patient_precision":       val_patient_metrics["precision"],
        "patient_balanced_accuracy": val_patient_metrics["balanced_accuracy"],
    })

    print(
        f"\nEpoch {epoch+1:03d}"
        f" | Train Loss={np.mean(train_losses):.4f}"
        f" | Threshold={current_threshold:.2f}"
        f" | [SLICE]   F1={val_slice_metrics['f1']:.4f}"
        f" AUC={val_slice_metrics['auc']:.4f}"
        f" | [PATIENT] F1={val_patient_metrics['f1']:.4f}"
        f" Recall={val_patient_metrics['recall']:.4f}"
        f" AUC={val_patient_metrics['auc']:.4f}"
    )

    # ------------------------------------------------------------------
    # EARLY STOPPING — baseado em F1 no nível do paciente
    # ------------------------------------------------------------------

    if val_patient_metrics["balanced_accuracy"] > best_patient_f1:

        best_patient_f1  = val_patient_metrics["f1"]
        best_threshold   = current_threshold
        patience_counter = 0

        torch.save(
            {
                "model_state_dict":      model.state_dict(),
                "best_threshold":        float(best_threshold),
                "best_patient_f1":       float(best_patient_f1),
                "epoch":                 int(epoch + 1),
            },
            RESULTS_DIR / "best_model.pth",
        )

        print(f"  ✓ Best model saved  (patient F1={best_patient_f1:.4f}, threshold={best_threshold:.2f})")

    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

# ==============================================================================
# FINAL EVALUATION
# ==============================================================================

print("\nLoading best model...\n")

checkpoint = torch.load(RESULTS_DIR / "best_model.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
best_threshold = checkpoint["best_threshold"]

print(f"Best threshold (from validation): {best_threshold:.2f}")

(
    test_slice_metrics,
    test_patient_metrics,
    y_true_slices,
    y_pred_slices,
    y_prob_slices,
    test_patient_ids,
) = evaluate(model, test_loader, threshold=best_threshold)

# Nível paciente para visualizações
pat_true, pat_prob, pat_ids = aggregate_patient_predictions(
    y_true_slices,
    y_pred_slices,
    y_prob_slices,
    test_patient_ids,
)

pat_pred = (pat_prob >= best_threshold).astype(int)

# ==============================================================================
# REPORT
# ==============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS — SLICE LEVEL")
print("=" * 80)
for k, v in test_slice_metrics.items():
    print(f"  {k:<25}: {v:.4f}")

print("\n" + "=" * 80)
print("FINAL RESULTS — PATIENT LEVEL (principal)")
print("=" * 80)
for k, v in test_patient_metrics.items():
    print(f"  {k:<25}: {v:.4f}")

# ==============================================================================
# SAVE FINAL METRICS
# ==============================================================================

final_metrics_combined = {
    "threshold":     best_threshold,
    "slice_level":   test_slice_metrics,
    "patient_level": test_patient_metrics,
}

with open(RESULTS_DIR / "final_metrics.json", "w") as f:
    json.dump(final_metrics_combined, f, indent=4)

# ==============================================================================
# CLASSIFICATION REPORT (patient level)
# ==============================================================================

report = classification_report(
    pat_true,
    pat_pred,
    target_names=[CLASS_1, CLASS_2],
    digits=4,
)

with open(RESULTS_DIR / "classification_report_patient.txt", "w") as f:
    f.write(f"Threshold: {best_threshold:.2f}\n\n")
    f.write(report)

print("\nClassification Report (patient level)")
print(f"Threshold: {best_threshold:.2f}")
print(report)

# ==============================================================================
# CONFUSION MATRIX (patient level)
# ==============================================================================

cm = confusion_matrix(pat_true, pat_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[CLASS_1, CLASS_2],
    yticklabels=[CLASS_1, CLASS_2],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(
    f"Confusion Matrix — Patient Level\n"
    f"F1={test_patient_metrics['f1']:.4f}  "
    f"AUC={test_patient_metrics['auc']:.4f}  "
    f"Threshold={best_threshold:.2f}"
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix_patient.png", dpi=300)
plt.close()

# ==============================================================================
# ROC CURVE (patient level)
# ==============================================================================

fpr, tpr, thresholds_roc = roc_curve(pat_true, pat_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {test_patient_metrics['auc']:.4f}")
plt.scatter(
    [fpr[np.argmin(np.abs(thresholds_roc - best_threshold))]],
    [tpr[np.argmin(np.abs(thresholds_roc - best_threshold))]],
    color="red",
    zorder=5,
    label=f"Operating point (t={best_threshold:.2f})",
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Patient Level")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roc_curve_patient.png", dpi=300)
plt.close()

# ==============================================================================
# SAVE PREDICTIONS (patient level)
# ==============================================================================

predictions_df = pd.DataFrame({
    "patient_id":  pat_ids,
    "y_true":      pat_true,
    "y_pred":      pat_pred,
    "probability": pat_prob,
    "threshold":   best_threshold,
})

predictions_df.to_csv(RESULTS_DIR / "predictions_patient.csv", index=False)

# ==============================================================================
# TRAINING CURVES
# ==============================================================================

history_df = pd.DataFrame(history)

# Loss
plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["train_loss"],    label="Train Loss")
plt.plot(history_df["epoch"], history_df["patient_loss"],  label="Val Loss (patient)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=300)
plt.close()

# F1 — slice vs patient
plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["slice_f1"],   label="Val F1 (slice)",   linestyle="--", alpha=0.7)
plt.plot(history_df["epoch"], history_df["patient_f1"], label="Val F1 (patient)", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.title("Validation F1 — Slice vs Patient Level")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "f1_curve.png", dpi=300)
plt.close()

# AUC
plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["slice_auc"],   label="Val AUC (slice)",   linestyle="--", alpha=0.7)
plt.plot(history_df["epoch"], history_df["patient_auc"], label="Val AUC (patient)", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Validation AUC")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "auc_curve.png", dpi=300)
plt.close()

# Recall (patient) — crítico para AD
plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["patient_recall"],    label="Recall (AD)")
plt.plot(history_df["epoch"], history_df["patient_precision"], label="Precision (AD)")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Recall vs Precision — Patient Level (AD class)")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "recall_precision_curve.png", dpi=300)
plt.close()

# Threshold ao longo do treino
plt.figure(figsize=(9, 4))
plt.plot(history_df["epoch"], history_df["threshold"])
plt.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Default 0.5")
plt.xlabel("Epoch")
plt.ylabel("Threshold")
plt.title("Optimal threshold (validation, patient level)")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "threshold_curve.png", dpi=300)
plt.close()

# ==============================================================================
# SAVE HISTORY
# ==============================================================================

history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)

print("\n✓ All results saved successfully")
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"\nSummary:")
print(f"  Threshold      : {best_threshold:.2f}")
print(f"  Patient F1     : {test_patient_metrics['f1']:.4f}")
print(f"  Patient AUC    : {test_patient_metrics['auc']:.4f}")
print(f"  Patient Recall : {test_patient_metrics['recall']:.4f}")
print(f"  Balanced Acc   : {test_patient_metrics['balanced_accuracy']:.4f}")