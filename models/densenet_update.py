"""
Treinamento de UM ÚNICO DenseNet-121 (pré-treinado ImageNet) para
classificação binária CN vs AD, usando o split train/test JÁ EXISTENTE
(sem cross-validation, sem random search, sem unir train e test).

Protocolo:
  1. Carrega train/ e test/ separadamente, com diagnóstico explícito em cada
     etapa (linhas no CSV -> depois do filtro CN/AD -> pastas encontradas em
     disco), para nunca mais perder pacientes silenciosamente.
  2. Reserva uma fatia pequena do TREINO (patient-level, estratificada) só
     para early stopping e calibração de threshold — nunca o test/.
  3. Treina um único modelo, com descongelamento em estágios (staged
     unfreezing) e learning rate por camada — técnica padrão de transfer
     learning, hiperparâmetros fixos (sem busca).
  4. Avalia UMA ÚNICA VEZ no test/ (nunca tocado antes), reportando:
     accuracy, precisão, recall, F1, matriz de confusão (+ AUC como extra).

Estrutura de dados esperada:
  Data/ADNI_FINAL_Updated/{train,test}/{CN,AD}/<subject_id>/*.npy
  Data/SPLITS_U/{train,test}_labels.csv  (colunas: subject_id, group)
"""

import json
import random
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
from sklearn.model_selection import train_test_split
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

RESULTS_DIR = Path(f"results_densenet_single_updated_{CLASS_1}_vs_{CLASS_2}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hiperparâmetros fixos — mesmos valores do protocolo original, já validado
# na TCC I (AUC 75,5% / balanced accuracy 73,0%). Sem busca de hiperparâmetro.
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCH_SCHEDULE = [10, 15, 15, 20]   # 4 estágios de descongelamento, 60 épocas no máximo
EARLY_STOPPING_PATIENCE = 10
VAL_FRACTION = 0.15                 # fatia do TREINO p/ early stopping e threshold (nunca o test/)
SELECTION_METRIC = "balanced_accuracy"  # critério único: early stopping, threshold e checkpoint


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
# CARREGAMENTO DE REGISTROS — COM DIAGNÓSTICO EXPLÍCITO EM CADA ETAPA
# ============================================================================

def load_records(labels_path, split_dir, split_name):
    print(f"\n--- Carregando '{split_name}' ---")
    print(f"  CSV de labels : {labels_path}")
    print(f"  Pasta de imagens: {split_dir}")

    if not Path(labels_path).exists():
        raise FileNotFoundError(f"CSV não encontrado: {labels_path}. Ajuste o caminho no topo do script.")
    if not Path(split_dir).exists():
        raise FileNotFoundError(f"Pasta não encontrada: {split_dir}. Ajuste o caminho no topo do script.")

    df = pd.read_csv(labels_path)
    print(f"  Total de linhas no CSV: {len(df)}")
    print(f"  Valores únicos em 'group': {sorted(df['group'].unique().tolist())}")

    df = df[df["group"].isin([CLASS_1, CLASS_2])]
    print(f"  Linhas após filtro {CLASS_1}/{CLASS_2}: {len(df)}")

    records = []
    missing = []
    for _, row in df.iterrows():
        patient_id = str(row["subject_id"])
        label = row["group"]
        patient_dir = Path(split_dir) / label / patient_id
        if patient_dir.exists():
            records.append({"patient_id": patient_id, "label": label, "patient_dir": patient_dir})
        else:
            missing.append((patient_id, label))

    print(f"  Pastas encontradas em disco: {len(records)} / {len(df)}")

    if missing:
        print(f"  AVISO: {len(missing)} paciente(s) do CSV NÃO têm pasta correspondente em disco.")
        print(f"  Exemplos não encontrados: {missing[:5]}")
        example_label = missing[0][1]
        example_folder = Path(split_dir) / example_label
        if example_folder.exists():
            real_names = sorted(p.name for p in example_folder.iterdir())
            print(f"  Conteúdo real de {example_folder} (5 primeiros): {real_names[:5]}")
        else:
            print(f"  A própria pasta {example_folder} não existe — confira CLASS_1/CLASS_2 "
                  f"e a estrutura de diretórios.")

    n_cn = sum(1 for r in records if r["label"] == CLASS_1)
    n_ad = sum(1 for r in records if r["label"] == CLASS_2)
    print(f"  Resultado final '{split_name}': {len(records)} pacientes ({n_cn} {CLASS_1} / {n_ad} {CLASS_2})")

    if len(records) < 0.9 * len(df):
        print(f"  *** ALERTA: mais de 10% dos pacientes do CSV não foram encontrados em disco. ***")
        print(f"  *** Confira TRAIN_DIR/TEST_DIR e os nomes de pasta antes de confiar nos resultados. ***")

    return records


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
        # Inclui somente parâmetros que foram descongelados.
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({
                "params": params,
                "lr": lr,
                "weight_decay": weight_decay,
            })

    features = model.features

    if stage == 1:
        # Apenas o classificador.
        add_group(model.classifier, base_lr)

    elif stage == 2:
        # Classificador + DenseBlock4 + Transition3.
        add_group(features.denseblock4, base_lr / 5)
        add_group(features.transition3, base_lr / 5)
        add_group(model.classifier, base_lr / 2)

    elif stage == 3:
        # Adiciona DenseBlock3 + Transition2.
        add_group(features.denseblock3, base_lr / 20)
        add_group(features.transition2, base_lr / 20)
        add_group(features.denseblock4, base_lr / 10)
        add_group(features.transition3, base_lr / 10)
        add_group(model.classifier, base_lr / 5)

    elif stage == 4:
        # Rede completa: todas as camadas treináveis entram no otimizador.
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

    # Verifica se todos os parâmetros treináveis foram incluídos.
    optimized_ids = {
        id(p)
        for group in param_groups
        for p in group["params"]
    }
    trainable_ids = {
        id(p)
        for p in model.parameters()
        if p.requires_grad
    }

    if optimized_ids != trainable_ids:
        raise RuntimeError(
            "Os parâmetros treináveis não correspondem aos parâmetros "
            "incluídos no otimizador."
        )

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
# TREINO
# ============================================================================

def main():
    seed_everything(SEED)

    all_train_records = load_records(TRAIN_LABELS, TRAIN_DIR, "train")
    test_records = load_records(TEST_LABELS, TEST_DIR, "test")

    train_labels = [r["label"] for r in all_train_records]
    train_records, val_records = train_test_split(
        all_train_records, test_size=VAL_FRACTION, stratify=train_labels, random_state=SEED
    )
    print(f"\nSplit final: {len(train_records)} treino | {len(val_records)} validação interna "
          f"(early stopping/threshold) | {len(test_records)} teste (held-out)")

    train_dataset = MRIDataset(train_records, transform=train_transform)
    val_dataset = MRIDataset(val_records, transform=eval_transform)
    test_dataset = MRIDataset(test_records, transform=eval_transform)
    print(f"Slices: {len(train_dataset)} treino | {len(val_dataset)} validação | {len(test_dataset)} teste")

    train_labels_numeric = [0 if r["label"] == CLASS_1 else 1 for r in train_records]
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=train_labels_numeric)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Pesos de classe (balanceamento via loss ponderada): {CLASS_1}={weights[0]:.3f}  {CLASS_2}={weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
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

    s1, s2, s3, s4 = EPOCH_SCHEDULE
    stage_boundaries = {s1: 2, s1 + s2: 3, s1 + s2 + s3: 4}

    best_score, best_threshold, patience_counter = -1.0, 0.5, 0
    history = []

    print(f"\n===== Iniciando treino (até {total_epochs} épocas, patience={EARLY_STOPPING_PATIENCE}) =====\n")

    for epoch in range(total_epochs):
        if epoch in stage_boundaries:
            new_stage = stage_boundaries[epoch]
            print(f"\n--- Estágio {new_stage}: descongelando mais camadas ---\n")
            unfreeze_stage(model, stage=new_stage)
            optimizer = get_layerwise_optimizer(model, BASE_LR, stage=new_stage, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - epoch, eta_min=1e-6
            )
            patience_counter = 0  # novo estágio merece chance justa antes do early stopping

        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
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

        val_metrics_05, pat_true, pat_prob, _, _ = evaluate(model, val_loader, criterion, threshold=best_threshold)
        current_threshold, current_score = find_best_threshold(pat_true, pat_prob)
        current_preds = (pat_prob >= current_threshold).astype(int)
        current_metrics = compute_metrics(pat_true, current_preds, pat_prob)

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_metrics_05["loss"],
            "threshold": current_threshold,
            "val_f1": current_metrics["f1"],
            "val_balanced_accuracy": current_metrics["balanced_accuracy"],
            "val_auc": current_metrics["auc"],
        })

        print(f"Epoch {epoch + 1:03d} | train_loss={np.mean(train_losses):.4f} | "
              f"val_{SELECTION_METRIC}={current_score:.4f} | threshold={current_threshold:.2f}")

        if current_score > best_score:
            best_score, best_threshold, patience_counter = current_score, current_threshold, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "best_threshold": float(best_threshold),
                 "best_score": float(best_score), "epoch": int(epoch + 1)},
                RESULTS_DIR / "best_model.pth",
            )
            print(f"  -> novo melhor modelo salvo (val_{SELECTION_METRIC}={best_score:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping na época {epoch + 1}")
            break

    pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)

    # ---------------- Avaliação final, uma única vez, no test/ ----------------
    print("\n===== Carregando melhor checkpoint e avaliando no TEST (held-out) =====")
    checkpoint = torch.load(RESULTS_DIR / "best_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["best_threshold"]
    print(f"Threshold (escolhido na validação interna, nunca no teste): {best_threshold:.2f}")

    test_metrics, y_true, y_prob, y_pred, patient_ids = evaluate(model, test_loader, criterion, threshold=best_threshold)

    print("\n===== RESULTADOS FINAIS (test/, nível paciente) =====")
    for k in ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]:
        print(f"  {k:<20}: {test_metrics[k]:.4f}")

    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump({"threshold": best_threshold, "n_patients_test": len(y_true), "metrics": test_metrics}, f, indent=4)

    report = classification_report(y_true, y_pred, target_names=[CLASS_1, CLASS_2], digits=4)
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Threshold: {best_threshold:.2f}\nN: {len(y_true)}\n\n{report}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[CLASS_1, CLASS_2], yticklabels=[CLASS_1, CLASS_2])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — Test Set (n={len(y_true)})\n"
              f"F1={test_metrics['f1']:.4f}  Acc={test_metrics['accuracy']:.4f}  Threshold={best_threshold:.2f}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    pd.DataFrame({
        "patient_id": patient_ids, "y_true": y_true, "y_pred": y_pred, "probability": y_prob,
    }).to_csv(RESULTS_DIR / "predictions_test.csv", index=False)

    print(f"\nTudo salvo em: {RESULTS_DIR}/")
    for fname in ["best_model.pth", "final_metrics.json", "classification_report.txt",
                  "confusion_matrix.png", "predictions_test.csv", "training_history.csv"]:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()