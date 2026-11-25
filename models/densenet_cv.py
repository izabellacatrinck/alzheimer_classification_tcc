"""
DenseNet-121 (pré-treinado ImageNet) para classificação binária CN vs AD com
VALIDAÇÃO CRUZADA ESTRATIFICADA POR PACIENTE (k folds).

Mesma receita do script de split único (hiperparâmetros, transforms, descongelamento
em estágios, learning rate por camada, agregação por paciente, critério de seleção),
mudando apenas o protocolo de avaliação:

  1. Junta train/ + test/ em um único conjunto de PACIENTES (1 linha por subject_id).
  2. StratifiedKFold (k=5) no nível do paciente: nenhum paciente aparece em dois
     folds, logo não há vazamento entre treino e teste.
  3. Em cada fold: 15% do treino do fold (estratificado, por paciente) vira
     validação interna para early stopping, escolha de threshold e checkpoint.
     O fold de teste só é usado UMA vez, no final.
  4. Agrega as predições out-of-fold (OOF) de todos os pacientes e reporta:
       - métricas por fold (média ± desvio padrão);
       - métricas agrupadas (pooled) com IC 95% por bootstrap de pacientes;
       - as mesmas métricas só nos pacientes do test/ original (para comparar
         de forma justa com o MedGemma, que foi avaliado nesses 96 pacientes);
       - tempo de inferência por paciente (9 fatias), para comparar com o MedGemma.

Estrutura de dados esperada (igual ao script anterior):
  Data/ADNI_FINAL_Updated/{train,test}/{CN,AD}/<subject_id>/*.npy
  Data/SPLITS_U/{train,test}_labels.csv  (colunas: subject_id, group)

Saídas em results_densenet_cv_CN_vs_AD/:
  fold_<k>/{best_model.pth, training_history.csv, final_metrics.json,
            classification_report.txt, confusion_matrix.png, predictions_test.csv}
  oof_predictions.csv, cv_fold_metrics.csv, cv_summary.json,
  confusion_matrix_pooled.png, confusion_matrix_original_test.png

Se a execução for interrompida, basta rodar de novo: folds já concluídos são
reaproveitados (fold_<k>/final_metrics.json + predictions_test.csv).
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121
from tqdm import tqdm

# ============================================================================
# CONFIGURAÇÃO GERAL
# ============================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
GRAD_CLIP = 1.0
USE_MIXED_PRECISION = True

CLASS_1 = "CN"  # label 0
CLASS_2 = "AD"  # label 1

# AJUSTE ESTES 4 CAMINHOS PARA A SUA ESTRUTURA REAL ANTES DE RODAR.
TRAIN_DIR = Path("Data/ADNI_FINAL_Updated/train")
TEST_DIR = Path("Data/ADNI_FINAL_Updated/test")
TRAIN_LABELS = Path("Data/SPLITS_U/train_labels.csv")
TEST_LABELS = Path("Data/SPLITS_U/test_labels.csv")

RESULTS_DIR = Path(f"results_densenet_cv_{CLASS_1}_vs_{CLASS_2}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hiperparâmetros fixos — idênticos ao script de split único.
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCH_SCHEDULE = [10, 15, 15, 20]   # 4 estágios de descongelamento, 60 épocas no máximo
EARLY_STOPPING_PATIENCE = 10
VAL_FRACTION = 0.15                 # fatia do treino de CADA fold (nunca o fold de teste)
SELECTION_METRIC = "balanced_accuracy"

# Validação cruzada / estatística
N_FOLDS = 5
N_BOOTSTRAP = 2000                  # reamostragens de pacientes para o IC 95%
N_TIME_PATIENTS = 10                # pacientes usados para medir o tempo de inferência por fold

METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.95, 1.0), ratio=(0.98, 1.02)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(4),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.99, 1.01)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================================
# DATASET
# ============================================================================

class MRIDataset(Dataset):
    def __init__(self, records, transform=None, max_slices=9):
        self.samples = []
        self.transform = transform
        self.label_map = {CLASS_1: 0, CLASS_2: 1}

        for record in records:
            patient_dir = record["patient_dir"]
            label = record["label"]

            npy_files = sorted(Path(patient_dir).glob("*.npy"))
            n = len(npy_files)
            if n <= max_slices:
                selected = npy_files
            else:
                center = n // 2
                half = max_slices // 2
                selected = npy_files[center - half: center + half]

            for npy_file in selected:
                self.samples.append({
                    "path": npy_file,
                    "label": label,
                    "patient_id": record["patient_id"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = np.load(sample["path"]).astype(np.float32)

        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        img = np.clip(img, p1, p99)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.label_map[sample["label"]]
        return img, label, sample["patient_id"]


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    patient_ids = [b[2] for b in batch]
    return images, labels, patient_ids


# ============================================================================
# CARREGAMENTO DE REGISTROS — COM DIAGNÓSTICO EXPLÍCITO
# ============================================================================

def load_records(labels_path, split_dir, split_name):
    print(f"\n--- Carregando '{split_name}' ---")
    print(f"  CSV de labels   : {labels_path}")
    print(f"  Pasta de imagens: {split_dir}")

    if not Path(labels_path).exists():
        raise FileNotFoundError(f"CSV não encontrado: {labels_path}. Ajuste o caminho no topo do script.")
    if not Path(split_dir).exists():
        raise FileNotFoundError(f"Pasta não encontrada: {split_dir}. Ajuste o caminho no topo do script.")

    df = pd.read_csv(labels_path)
    print(f"  Total de linhas no CSV: {len(df)}")
    df = df[df["group"].isin([CLASS_1, CLASS_2])]
    print(f"  Linhas após filtro {CLASS_1}/{CLASS_2}: {len(df)}")

    records, missing = [], []
    for _, row in df.iterrows():
        patient_id = str(row["subject_id"])
        label = row["group"]
        patient_dir = Path(split_dir) / label / patient_id
        if patient_dir.exists():
            records.append({
                "patient_id": patient_id,
                "label": label,
                "patient_dir": patient_dir,
                "original_split": split_name,
            })
        else:
            missing.append((patient_id, label))

    print(f"  Pastas encontradas em disco: {len(records)} / {len(df)}")
    if missing:
        print(f"  AVISO: {len(missing)} paciente(s) do CSV NÃO têm pasta em disco. Exemplos: {missing[:5]}")

    n_cn = sum(1 for r in records if r["label"] == CLASS_1)
    n_ad = sum(1 for r in records if r["label"] == CLASS_2)
    print(f"  Resultado final '{split_name}': {len(records)} pacientes ({n_cn} {CLASS_1} / {n_ad} {CLASS_2})")

    if len(records) < 0.9 * len(df):
        print("  *** ALERTA: mais de 10% dos pacientes do CSV não foram encontrados em disco. ***")

    return records


def build_patient_pool():
    """Une train + test em um único conjunto de pacientes (1 registro por subject_id)."""
    train_records = load_records(TRAIN_LABELS, TRAIN_DIR, "train")
    test_records = load_records(TEST_LABELS, TEST_DIR, "test")

    pool, seen = [], {}
    for r in train_records + test_records:
        pid = r["patient_id"]
        if pid in seen:
            print(f"  AVISO: paciente {pid} aparece em mais de um split; mantendo a primeira ocorrência "
                  f"({seen[pid]['original_split']}).")
            if seen[pid]["label"] != r["label"]:
                raise ValueError(f"Paciente {pid} com rótulos conflitantes entre splits.")
            continue
        seen[pid] = r
        pool.append(r)

    n_cn = sum(1 for r in pool if r["label"] == CLASS_1)
    n_ad = sum(1 for r in pool if r["label"] == CLASS_2)
    print(f"\n=== Conjunto total para CV: {len(pool)} pacientes ({n_cn} {CLASS_1} / {n_ad} {CLASS_2}) ===")
    return pool


# ============================================================================
# MODELO / OTIMIZADOR / DESCONGELAMENTO EM ESTÁGIOS
# ============================================================================

def build_model():
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(1024, 2)
    return model.to(DEVICE)


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


def get_layerwise_optimizer(model, base_lr, stage, weight_decay):
    param_groups = []

    def add_group(module, lr):
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})

    features = model.features

    if stage == 1:
        add_group(model.classifier, base_lr)

    elif stage == 2:
        add_group(features.denseblock4, base_lr / 5)
        add_group(features.transition3, base_lr / 5)
        add_group(model.classifier, base_lr / 2)

    elif stage == 3:
        add_group(features.denseblock3, base_lr / 20)
        add_group(features.transition2, base_lr / 20)
        add_group(features.denseblock4, base_lr / 10)
        add_group(features.transition3, base_lr / 10)
        add_group(model.classifier, base_lr / 5)

    elif stage == 4:
        add_group(features.conv0, base_lr / 100)
        add_group(features.norm0, base_lr / 100)
        add_group(features.denseblock1, base_lr / 100)
        add_group(features.transition1, base_lr / 100)
        add_group(features.denseblock2, base_lr / 50)
        add_group(features.transition2, base_lr / 50)
        add_group(features.denseblock3, base_lr / 20)
        add_group(features.transition3, base_lr / 10)
        add_group(features.denseblock4, base_lr / 10)
        add_group(features.norm5, base_lr / 10)
        add_group(model.classifier, base_lr / 5)

    else:
        raise ValueError(f"Estágio inválido: {stage}")

    optimized_ids = {id(p) for g in param_groups for p in g["params"]}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if optimized_ids != trainable_ids:
        raise RuntimeError("Os parâmetros treináveis não correspondem aos do otimizador.")

    return optim.AdamW(param_groups)


# ============================================================================
# MÉTRICAS / AGREGAÇÃO POR PACIENTE / THRESHOLD
# ============================================================================

def aggregate_patient_predictions(all_targets, all_probs, all_patient_ids):
    patient_probs = defaultdict(list)
    patient_targets = {}
    for pid, target, prob in zip(all_patient_ids, all_targets, all_probs):
        patient_probs[pid].append(prob)
        patient_targets[pid] = target
    pids = list(patient_probs.keys())
    agg_probs = np.array([np.mean(patient_probs[pid]) for pid in pids])
    true_labels = np.array([patient_targets[pid] for pid in pids])
    return true_labels, agg_probs, pids


def find_best_threshold(y_true, y_prob, metric=SELECTION_METRIC):
    thresholds = np.arange(0.20, 0.80, 0.01)
    best_thresh, best_score = 0.5, -1.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = (balanced_accuracy_score(y_true, preds) if metric == "balanced_accuracy"
                 else f1_score(y_true, preds, zero_division=0))
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["auc"] = 0.0
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    losses, all_probs, all_targets, all_patient_ids = [], [], [], []

    for images, targets, patient_ids in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
            outputs = model(images)
            loss = criterion(outputs.float(), targets)
            probs = torch.softmax(outputs, dim=1)[:, 1]

        losses.append(loss.item())
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_patient_ids.extend(patient_ids)

    all_targets, all_probs = np.array(all_targets), np.array(all_probs)
    pat_true, pat_prob, pat_ids = aggregate_patient_predictions(all_targets, all_probs, all_patient_ids)
    pat_pred = (pat_prob >= threshold).astype(int)
    patient_metrics = compute_metrics(pat_true, pat_pred, pat_prob)
    patient_metrics["loss"] = float(np.mean(losses))

    return patient_metrics, pat_true, pat_prob, pat_pred, pat_ids


# ============================================================================
# TEMPO DE INFERÊNCIA POR PACIENTE (9 fatias em um único forward)
# ============================================================================

def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_inference_time(model, records, n_patients=N_TIME_PATIENTS):
    """Mede, por paciente: (a) pipeline completo (carregar .npy + pré-processar + forward)
    e (b) apenas o forward. O primeiro paciente é descartado como aquecimento."""
    model.eval()
    subset = records[: n_patients + 1]
    full_times, forward_times = [], []

    for i, rec in enumerate(subset):
        ds = MRIDataset([rec], transform=eval_transform)
        _sync()
        t0 = time.perf_counter()
        batch = torch.stack([ds[j][0] for j in range(len(ds))]).to(DEVICE)
        _sync()
        t1 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
            _ = model(batch)
        _sync()
        t2 = time.perf_counter()
        if i == 0:
            continue  # aquecimento
        full_times.append(t2 - t0)
        forward_times.append(t2 - t1)

    return {
        "n_patients_measured": len(full_times),
        "mean_seconds_per_patient_full": float(np.mean(full_times)),
        "std_seconds_per_patient_full": float(np.std(full_times)),
        "mean_seconds_per_patient_forward": float(np.mean(forward_times)),
        "std_seconds_per_patient_forward": float(np.std(forward_times)),
    }


# ============================================================================
# UM FOLD: TREINO + AVALIAÇÃO NO FOLD DE TESTE
# ============================================================================

def run_fold(fold, trainval_records, test_records):
    fold_dir = RESULTS_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = fold_dir / "final_metrics.json"
    preds_path = fold_dir / "predictions_test.csv"
    if metrics_path.exists() and preds_path.exists():
        print(f"\n>>> Fold {fold}: já concluído, reaproveitando resultados salvos.")
        with open(metrics_path) as f:
            saved = json.load(f)
        return pd.read_csv(preds_path, dtype={"patient_id": str}), saved

    fold_seed = SEED + fold
    seed_everything(fold_seed)

    labels = [r["label"] for r in trainval_records]
    train_records, val_records = train_test_split(
        trainval_records, test_size=VAL_FRACTION, stratify=labels, random_state=fold_seed
    )
    print(f"\n{'=' * 70}\nFOLD {fold}/{N_FOLDS}: {len(train_records)} treino | "
          f"{len(val_records)} validação interna | {len(test_records)} teste\n{'=' * 70}")

    train_dataset = MRIDataset(train_records, transform=train_transform)
    val_dataset = MRIDataset(val_records, transform=eval_transform)
    test_dataset = MRIDataset(test_records, transform=eval_transform)
    print(f"Slices: {len(train_dataset)} treino | {len(val_dataset)} validação | {len(test_dataset)} teste")

    y_train = [0 if r["label"] == CLASS_1 else 1 for r in train_records]
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    g = torch.Generator()
    g.manual_seed(fold_seed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    model = build_model()
    unfreeze_stage(model, stage=1)
    optimizer = get_layerwise_optimizer(model, BASE_LR, stage=1, weight_decay=WEIGHT_DECAY)

    total_epochs = sum(EPOCH_SCHEDULE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

    s1, s2, s3, _ = EPOCH_SCHEDULE
    stage_boundaries = {s1: 2, s1 + s2: 3, s1 + s2 + s3: 4}

    best_score, best_threshold, patience_counter = -1.0, 0.5, 0
    history = []

    for epoch in range(total_epochs):
        if epoch in stage_boundaries:
            new_stage = stage_boundaries[epoch]
            print(f"\n--- [Fold {fold}] Estágio {new_stage}: descongelando mais camadas ---\n")
            unfreeze_stage(model, stage=new_stage)
            optimizer = get_layerwise_optimizer(model, BASE_LR, stage=new_stage, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - epoch, eta_min=1e-6
            )
            patience_counter = 0

        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch + 1}/{total_epochs}")
        for images, targets, _ in loop:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs.float(), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            loop.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        val_metrics, pat_true, pat_prob, _, _ = evaluate(model, val_loader, criterion, threshold=best_threshold)
        current_threshold, current_score = find_best_threshold(pat_true, pat_prob)
        current_preds = (pat_prob >= current_threshold).astype(int)
        current_metrics = compute_metrics(pat_true, current_preds, pat_prob)

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_metrics["loss"],
            "threshold": current_threshold,
            "val_f1": current_metrics["f1"],
            "val_balanced_accuracy": current_metrics["balanced_accuracy"],
            "val_auc": current_metrics["auc"],
        })

        print(f"[Fold {fold}] Epoch {epoch + 1:03d} | train_loss={np.mean(train_losses):.4f} | "
              f"val_{SELECTION_METRIC}={current_score:.4f} | threshold={current_threshold:.2f}")

        if current_score > best_score:
            best_score, best_threshold, patience_counter = current_score, current_threshold, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "best_threshold": float(best_threshold),
                 "best_score": float(best_score), "epoch": int(epoch + 1)},
                fold_dir / "best_model.pth",
            )
            print(f"  -> novo melhor modelo salvo (val_{SELECTION_METRIC}={best_score:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n[Fold {fold}] Early stopping na época {epoch + 1}")
            break

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    # ---------------- Avaliação no fold de teste (uma única vez) ----------------
    checkpoint = torch.load(fold_dir / "best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["best_threshold"]
    print(f"\n[Fold {fold}] Melhor época: {checkpoint['epoch']} | threshold (validação interna): {best_threshold:.2f}")

    test_metrics, y_true, y_prob, y_pred, patient_ids = evaluate(
        model, test_loader, criterion, threshold=best_threshold
    )

    print(f"\n===== [Fold {fold}] RESULTADOS (fold de teste, nível paciente) =====")
    for k in METRIC_KEYS:
        print(f"  {k:<20}: {test_metrics[k]:.4f}")

    timing = measure_inference_time(model, test_records)
    print(f"  tempo/paciente (pipeline completo): {timing['mean_seconds_per_patient_full']:.3f} s | "
          f"(só forward): {timing['mean_seconds_per_patient_forward']:.3f} s")

    saved = {
        "fold": fold,
        "best_epoch": int(checkpoint["epoch"]),
        "threshold": float(best_threshold),
        "n_patients_test": int(len(y_true)),
        "metrics": test_metrics,
        "inference_time": timing,
    }
    with open(metrics_path, "w") as f:
        json.dump(saved, f, indent=4)

    report = classification_report(y_true, y_pred, target_names=[CLASS_1, CLASS_2], digits=4, zero_division=0)
    with open(fold_dir / "classification_report.txt", "w") as f:
        f.write(f"Fold: {fold}\nThreshold: {best_threshold:.2f}\nN: {len(y_true)}\n\n{report}")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plot_confusion(cm, f"Fold {fold} — matriz de confusão (n={len(y_true)})", fold_dir / "confusion_matrix.png")

    preds_df = pd.DataFrame({
        "patient_id": patient_ids, "y_true": y_true, "y_pred": y_pred, "probability": y_prob,
    })
    preds_df.to_csv(preds_path, index=False)

    # libera memória da GPU antes do próximo fold
    del model, optimizer, scaler
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return preds_df, saved


# ============================================================================
# AGREGAÇÃO: MÉTRICAS POR FOLD, POOLED, BOOTSTRAP
# ============================================================================

def plot_confusion(cm, title, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[CLASS_1, CLASS_2], yticklabels=[CLASS_1, CLASS_2])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=N_BOOTSTRAP, seed=SEED):
    """IC 95% (percentil) por bootstrap de PACIENTES sobre as predições OOF."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, y_pred[idx], y_prob[idx])
        for k in METRIC_KEYS:
            samples[k].append(m[k])
    point = compute_metrics(y_true, y_pred, y_prob)
    return {
        k: {
            "value": float(point[k]),
            "ci95_low": float(np.percentile(samples[k], 2.5)),
            "ci95_high": float(np.percentile(samples[k], 97.5)),
        }
        for k in METRIC_KEYS
    }


def summarize_subset(df, name):
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    y_prob = df["probability"].to_numpy()
    ci = bootstrap_ci(y_true, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n----- {name} (n={len(df)}) — pooled com IC 95% bootstrap -----")
    for k in METRIC_KEYS:
        v = ci[k]
        print(f"  {k:<20}: {v['value']:.4f}  [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}]")
    return {"n_patients": int(len(df)), "metrics": ci, "confusion_matrix": cm.tolist()}, cm


def main():
    seed_everything(SEED)
    print(f"Dispositivo: {DEVICE}")

    pool = build_patient_pool()
    labels = np.array([0 if r["label"] == CLASS_1 else 1 for r in pool])
    split_map = {r["patient_id"]: r["original_split"] for r in pool}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_parts, fold_results = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(pool)), labels), start=1):
        trainval_records = [pool[i] for i in tr_idx]
        test_records = [pool[i] for i in te_idx]

        # garantia explícita de que não há paciente em comum
        assert not ({r["patient_id"] for r in trainval_records} & {r["patient_id"] for r in test_records})

        preds_df, saved = run_fold(fold, trainval_records, test_records)
        preds_df = preds_df.copy()
        preds_df["fold"] = fold
        oof_parts.append(preds_df)
        fold_results.append(saved)

    # ---------------- Predições OOF ----------------
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["patient_id"] = oof["patient_id"].astype(str)
    oof["original_split"] = oof["patient_id"].map(split_map)
    oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    assert oof["patient_id"].is_unique, "Paciente repetido nas predições OOF!"
    print(f"\nPredições OOF: {len(oof)} pacientes (cada um avaliado exatamente uma vez).")

    # ---------------- Métricas por fold ----------------
    fold_rows = []
    for r in fold_results:
        row = {"fold": r["fold"], "n_patients": r["n_patients_test"], "threshold": r["threshold"],
               "best_epoch": r["best_epoch"]}
        row.update({k: r["metrics"][k] for k in METRIC_KEYS})
        row["sec_per_patient_full"] = r["inference_time"]["mean_seconds_per_patient_full"]
        row["sec_per_patient_forward"] = r["inference_time"]["mean_seconds_per_patient_forward"]
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)

    print("\n===== Métricas por fold (fold de teste, nível paciente) =====")
    print(fold_df.round(4).to_string(index=False))

    per_fold_summary = {
        k: {"mean": float(fold_df[k].mean()), "std": float(fold_df[k].std(ddof=1))} for k in METRIC_KEYS
    }
    print("\n===== Média ± desvio padrão entre folds =====")
    for k in METRIC_KEYS:
        print(f"  {k:<20}: {per_fold_summary[k]['mean']:.4f} ± {per_fold_summary[k]['std']:.4f}")

    # ---------------- Pooled + bootstrap ----------------
    pooled, cm_pooled = summarize_subset(oof, f"TODOS os pacientes ({N_FOLDS}-fold CV)")
    plot_confusion(cm_pooled, f"Matriz de confusão agrupada — {N_FOLDS}-fold CV (n={len(oof)})",
                   RESULTS_DIR / "confusion_matrix_pooled.png")

    original_test = oof[oof["original_split"] == "test"]
    orig_summary, cm_orig = (None, None)
    if len(original_test) > 0:
        orig_summary, cm_orig = summarize_subset(
            original_test, "Somente pacientes do test/ ORIGINAL (comparável ao MedGemma)"
        )
        plot_confusion(cm_orig, f"Matriz de confusão — test/ original dentro da CV (n={len(original_test)})",
                       RESULTS_DIR / "confusion_matrix_original_test.png")

    summary = {
        "protocol": {
            "n_folds": N_FOLDS,
            "split": "StratifiedKFold por paciente (train + test unidos)",
            "val_fraction_inside_each_train_fold": VAL_FRACTION,
            "selection_metric": SELECTION_METRIC,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "n_patients_total": int(len(oof)),
        },
        "per_fold_mean_std": per_fold_summary,
        "pooled_all_patients": pooled,
        "pooled_original_test_patients": orig_summary,
        "inference_time": {
            "mean_seconds_per_patient_full": float(fold_df["sec_per_patient_full"].mean()),
            "mean_seconds_per_patient_forward": float(fold_df["sec_per_patient_forward"].mean()),
            "note": "9 fatias por paciente em um único forward; primeiro paciente de cada fold descartado como aquecimento",
        },
    }
    with open(RESULTS_DIR / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nTudo salvo em: {RESULTS_DIR}/")
    for fname in ["oof_predictions.csv", "cv_fold_metrics.csv", "cv_summary.json",
                  "confusion_matrix_pooled.png", "confusion_matrix_original_test.png"]:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()