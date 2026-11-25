"""
Avalia o MedGemma-4B-IT em modo FEW-SHOT (K exemplos de cada classe no
prompt, K=3 por padrão) para classificação binária CN vs AD —
metodologicamente comparável ao train_densenet_single.py e ao
evaluate_medgemma_one_shot.py: mesmo split train/test (sem
cross-validation), mesma fração de validação interna para calibrar
threshold, avaliação final única no test/.

FORMALIZAÇÃO DO PROMPT (mesmas fontes do one-shot, agora usando a parte
few-shot da literatura):

1. Estrutura: Ferber et al., "In-context learning enables multimodal
   large language models to classify cancer pathology images", Nature
   Communications 2024 (arXiv:2403.07407). Testam explicitamente k=1, 3,
   5 e 10 com um VLM generativo (GPT-4V) em classificação binária de
   imagens médicas. Usamos k=3 aqui — o primeiro degrau "few-shot" de
   fato no estudo deles, equilibrando ganho esperado (eles relatam
   melhora substancial saindo de poucos exemplos) com o custo
   computacional de processar múltiplas imagens por chamada numa GPU de
   6GB. Mantêm a mesma técnica de ALTERNAR entre as classes ao apresentar
   os exemplos ("cycling through the entire set of labels k times") em
   vez de apresentá-los em blocos — é a ordem reproduzida aqui.

2. Resposta forçada (Yes/No) + leitura dos logits do primeiro token:
   Farina et al. (2026, arXiv:2606.20852), mesma técnica já usada no
   zero-shot e no one-shot, para manter os três experimentos comparáveis
   entre si (mesmo mecanismo de extração de probabilidade).

3. Âncora de critérios radiológicos: Goyal et al. (2026, arXiv:2606.21861)
   + vocabulário consistente com a Tabela 2 do MedSigLIP na sua TCC I.

Desenho experimental (decisão específica deste script): o exemplo nº1 de
cada classe é reaproveitado do evaluate_medgemma_one_shot.py (lido de
results_medgemma_oneshot_CN_vs_AD/exemplars_used.json, se existir) — isso
isola o efeito de "mais exemplos" do efeito de "exemplos diferentes",
tornando zero-shot -> one-shot -> few-shot uma progressão limpa. Os 2
exemplos restantes por classe são sorteados de train_fit com seed fixa.

Protocolo de split (idêntico aos outros dois scripts, mesmo SEED):
  train/ -> train_fit (exemplos few-shot) / val_internal (calibra o
  threshold, nunca usado como exemplo) -> test/ (avaliado uma única vez).

Custo computacional: cada slice processa 2*K_SHOTS*N_SLICES_PER_EXEMPLAR + 1
imagens numa única chamada (com K_SHOTS=3 e N_SLICES_PER_EXEMPLAR=3: 19
imagens) — bem mais caro que o one-shot (3 imagens) e que a versão
anterior deste script (7 imagens, 1 slice por exemplo). Rode
--limit 3 --debug primeiro, e considere reduzir --slices-per-exemplar ou
--k-shots se a GPU não aguentar.

Requisitos:
    pip install transformers accelerate bitsandbytes torch pillow pandas numpy tqdm scikit-learn matplotlib seaborn

Uso:
    python evaluate_medgemma_few_shot.py --limit 3 --debug
    python evaluate_medgemma_few_shot.py
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
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
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# ============================================================================
# CONFIGURAÇÃO — mantenha idêntica ao train_densenet_single.py / one-shot
# ============================================================================

SEED = 42
MODEL_ID = "google/medgemma-4b-it"

CLASS_1 = "CN"  # label 0
CLASS_2 = "AD"  # label 1

TRAIN_DIR = Path("Data/ADNI_FINAL_Updated/train")
TEST_DIR = Path("Data/ADNI_FINAL_Updated/test")
TRAIN_LABELS = Path("Data/SPLITS_U/train_labels.csv")
TEST_LABELS = Path("Data/SPLITS_U/test_labels.csv")

ONE_SHOT_EXEMPLARS_FILE = Path("results_medgemma_oneshot_CN_vs_AD/exemplars_used.json")
K_SHOTS = 3  # exemplos (pacientes) POR CLASSE
N_SLICES_PER_EXEMPLAR = 3  # slices por paciente-exemplo (total de imagens de
                            # exemplo = 2 * K_SHOTS * N_SLICES_PER_EXEMPLAR)

RESULTS_DIR = Path(f"results_medgemma_fewshot_{CLASS_1}_vs_{CLASS_2}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRACTION = 0.15
MAX_SLICES_PER_PATIENT = 10
SELECTION_METRIC = "balanced_accuracy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

N_INFERENCE_WARMUP = 2
N_INFERENCE_MEASURE = 5  # few-shot processa 2*K_SHOTS*N_SLICES_PER_EXEMPLAR+1 imagens/slice — orçamento bem menor que o one-shot

ANSWER_TOKEN_AD = " Yes"
ANSWER_TOKEN_CN = " No"

RESEARCH_CONTEXT = (
    "This is part of an academic research study on automated Alzheimer's "
    "disease classification using the de-identified ADNI neuroimaging "
    "dataset. You are acting as an expert neuroradiologist assisting "
    "with a structured research classification task — this is a "
    "scientific evaluation of model performance on a research "
    "benchmark, not a clinical diagnosis for an individual patient."
)
CN_EXEMPLAR_TEXT_TEMPLATE = (
    "Example {n} ({n_slices} MRI slices from one patient): this patient "
    "has preserved hippocampal volume and no neurodegenerative "
    "abnormalities (Cognitively Normal)."
)
AD_EXEMPLAR_TEXT_TEMPLATE = (
    "Example {n} ({n_slices} MRI slices from one patient): this patient "
    "has hippocampal atrophy and enlarged ventricles consistent with "
    "Alzheimer's disease."
)
QUERY_PROMPT = (
    "Now consider the following new patient slice. Established "
    "structural markers of Alzheimer's disease on MRI include "
    "hippocampal atrophy, ventricular enlargement, and cortical "
    "thinning, particularly in the medial temporal lobe. Comparing this "
    "slice to the examples above, and considering these features, does "
    "this slice show signs of Alzheimer's disease? Answer with only Yes "
    "or No."
)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# CARREGAMENTO DE IMAGENS / REGISTROS — idêntico aos outros scripts
# ============================================================================

def select_slices(patient_dir, max_slices=MAX_SLICES_PER_PATIENT):
    npy_files = sorted(Path(patient_dir).glob("*.npy"))
    n = len(npy_files)
    if n <= max_slices:
        return npy_files
    center = n // 2
    half = max_slices // 2
    return npy_files[center - half: center + half]


def load_slice_as_image(npy_path):
    img = np.load(npy_path).astype(np.float32)
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def load_records(labels_path, split_dir, split_name):
    print(f"\n--- Carregando '{split_name}' ---")
    print(f"  CSV de labels : {labels_path}")
    print(f"  Pasta de imagens: {split_dir}")

    if not Path(labels_path).exists():
        raise FileNotFoundError(f"CSV não encontrado: {labels_path}.")
    if not Path(split_dir).exists():
        raise FileNotFoundError(f"Pasta não encontrada: {split_dir}.")

    df = pd.read_csv(labels_path)
    print(f"  Total de linhas no CSV: {len(df)}")
    print(f"  Valores únicos em 'group': {sorted(df['group'].unique().tolist())}")

    df = df[df["group"].isin([CLASS_1, CLASS_2])]
    print(f"  Linhas após filtro {CLASS_1}/{CLASS_2}: {len(df)}")

    records, missing = [], []
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
        print(f"  AVISO: {len(missing)} paciente(s) sem pasta correspondente. Exemplos: {missing[:5]}")

    n_cn = sum(1 for r in records if r["label"] == CLASS_1)
    n_ad = sum(1 for r in records if r["label"] == CLASS_2)
    print(f"  Resultado final '{split_name}': {len(records)} pacientes ({n_cn} {CLASS_1} / {n_ad} {CLASS_2})")

    if len(records) < 0.9 * len(df):
        print("  *** ALERTA: mais de 10% dos pacientes do CSV não foram encontrados em disco. ***")

    return records


# ============================================================================
# MODELO
# ============================================================================

def load_model_and_processor(quantize_4bit=True):
    print(f"Carregando {MODEL_ID} ...")
    if quantize_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, quantization_config=quant_config, device_map="auto",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def get_answer_token_ids(processor):
    tok_ad = processor.tokenizer.encode(ANSWER_TOKEN_AD, add_special_tokens=False)
    tok_cn = processor.tokenizer.encode(ANSWER_TOKEN_CN, add_special_tokens=False)
    if len(tok_ad) != 1 or len(tok_cn) != 1:
        print(f"AVISO: tokens candidatos não são de 1 único token (AD={tok_ad}, CN={tok_cn}).")
    id_ad, id_cn = tok_ad[0], tok_cn[0]
    print(f"Confirmação: id {id_ad} -> {processor.tokenizer.decode([id_ad])!r} | "
          f"id {id_cn} -> {processor.tokenizer.decode([id_cn])!r}")
    return id_ad, id_cn


# ============================================================================
# EXEMPLOS FEW-SHOT (K por classe, nº1 reaproveitado do one-shot se existir)
# ============================================================================

def select_exemplars(train_fit_records, k_shots=K_SHOTS, seed=SEED,
                      one_shot_exemplars_file=ONE_SHOT_EXEMPLARS_FILE):
    cn_candidates = [r for r in train_fit_records if r["label"] == CLASS_1]
    ad_candidates = [r for r in train_fit_records if r["label"] == CLASS_2]
    rng = random.Random(seed)

    cn_selected, ad_selected = [], []

    if one_shot_exemplars_file.exists():
        with open(one_shot_exemplars_file) as f:
            prev = json.load(f)
        cn_first = next((r for r in cn_candidates if r["patient_id"] == prev.get("cn_exemplar")), None)
        ad_first = next((r for r in ad_candidates if r["patient_id"] == prev.get("ad_exemplar")), None)
        if cn_first:
            cn_selected.append(cn_first)
        if ad_first:
            ad_selected.append(ad_first)
        if cn_first or ad_first:
            print(f"Reaproveitando exemplo(s) do one-shot como shot #1: "
                  f"CN={prev.get('cn_exemplar')}  AD={prev.get('ad_exemplar')}")
    else:
        print(f"Arquivo do one-shot não encontrado ({one_shot_exemplars_file}) — "
              f"sorteando {k_shots} exemplos novos por classe, sem reaproveitar nada.")

    cn_selected_ids = {r["patient_id"] for r in cn_selected}
    ad_selected_ids = {r["patient_id"] for r in ad_selected}
    cn_remaining = [r for r in cn_candidates if r["patient_id"] not in cn_selected_ids]
    ad_remaining = [r for r in ad_candidates if r["patient_id"] not in ad_selected_ids]
    rng.shuffle(cn_remaining)
    rng.shuffle(ad_remaining)

    while len(cn_selected) < k_shots and cn_remaining:
        cn_selected.append(cn_remaining.pop())
    while len(ad_selected) < k_shots and ad_remaining:
        ad_selected.append(ad_remaining.pop())

    return cn_selected[:k_shots], ad_selected[:k_shots]


def representative_slice_images(record, n_slices=N_SLICES_PER_EXEMPLAR, max_slices=MAX_SLICES_PER_PATIENT):
    """Retorna n_slices imagens desse paciente-exemplo, espaçadas igualmente
    dentro do conjunto de slices centrais já selecionado (não as n_slices
    mais próximas entre si, para dar alguma diversidade visual ao exemplo)."""
    slice_paths = select_slices(record["patient_dir"], max_slices=max_slices)
    if len(slice_paths) <= n_slices:
        chosen = slice_paths
    else:
        indices = sorted(set(np.linspace(0, len(slice_paths) - 1, n_slices).round().astype(int)))
        chosen = [slice_paths[i] for i in indices]
    return [load_slice_as_image(p) for p in chosen]


# ============================================================================
# SCORING FEW-SHOT
# ============================================================================

def build_few_shot_content(cn_image_sets, ad_image_sets, query_image):
    """
    cn_image_sets / ad_image_sets: lista de listas — cada elemento é o
    conjunto de N_SLICES_PER_EXEMPLAR imagens de UM paciente-exemplo (não
    uma imagem por "shot", agora são várias imagens por paciente-exemplo).
    Alterna entre as classes ao apresentar os exemplos (CN, AD, CN, AD,
    ...), seguindo Ferber et al. — não em blocos por classe. Dentro de
    cada exemplo, as N_SLICES_PER_EXEMPLAR imagens daquele paciente vêm
    juntas, sob a mesma legenda.
    """
    content = [{"type": "text", "text": RESEARCH_CONTEXT}]
    n = max(len(cn_image_sets), len(ad_image_sets))
    example_num = 1
    for i in range(n):
        if i < len(cn_image_sets):
            content.append({"type": "text", "text": CN_EXEMPLAR_TEXT_TEMPLATE.format(
                n=example_num, n_slices=len(cn_image_sets[i]))})
            for img in cn_image_sets[i]:
                content.append({"type": "image", "image": img})
            example_num += 1
        if i < len(ad_image_sets):
            content.append({"type": "text", "text": AD_EXEMPLAR_TEXT_TEMPLATE.format(
                n=example_num, n_slices=len(ad_image_sets[i]))})
            for img in ad_image_sets[i]:
                content.append({"type": "image", "image": img})
            example_num += 1
    content.append({"type": "text", "text": QUERY_PROMPT})
    content.append({"type": "image", "image": query_image})
    return content


@torch.inference_mode()
def score_slice_few_shot(model, processor, query_image, cn_image_sets, ad_image_sets, id_ad, id_cn, verbose=False):
    content = build_few_shot_content(cn_image_sets, ad_image_sets, query_image)
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=DTYPE)

    outputs = model.generate(
        **inputs, max_new_tokens=1, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )
    logits = outputs.scores[0][0]
    logit_ad, logit_cn = logits[id_ad].item(), logits[id_cn].item()
    prob_ad = torch.softmax(torch.tensor([logit_ad, logit_cn]), dim=0)[0].item()

    if verbose:
        top_id = int(torch.argmax(logits).item())
        top_token = processor.tokenizer.decode([top_id])
        print(f"      [debug] logit(Yes)={logit_ad:+.3f}  logit(No)={logit_cn:+.3f}  "
              f"P(AD)={prob_ad:.4f}  |  token mais provável: {top_token!r}")

    return prob_ad


def evaluate_patient(model, processor, patient_dir, cn_image_sets, ad_image_sets, id_ad, id_cn,
                      max_slices=MAX_SLICES_PER_PATIENT, verbose=False):
    slice_paths = select_slices(patient_dir, max_slices=max_slices)
    slice_probs = [
        score_slice_few_shot(model, processor, load_slice_as_image(p), cn_image_sets, ad_image_sets,
                              id_ad, id_cn, verbose=verbose)
        for p in slice_paths
    ]
    return float(np.mean(slice_probs)), len(slice_probs)


def evaluate_population(model, processor, records, cn_image_sets, ad_image_sets, id_ad, id_cn,
                         max_slices, verbose_first_n=0, desc="Avaliando"):
    rows = []
    label_map = {CLASS_1: 0, CLASS_2: 1}
    for i, record in enumerate(tqdm(records, desc=desc)):
        verbose = i < verbose_first_n
        prob_ad, n_slices = evaluate_patient(model, processor, record["patient_dir"], cn_image_sets, ad_image_sets,
                                              id_ad, id_cn, max_slices=max_slices, verbose=verbose)
        rows.append({
            "patient_id": record["patient_id"], "y_true": label_map[record["label"]],
            "probability": prob_ad, "n_slices": n_slices,
        })
        if verbose:
            print(f"  {record['patient_id']} (label={record['label']}) -> P(AD)={prob_ad:.4f}")
    return pd.DataFrame(rows)


# ============================================================================
# MÉTRICAS / THRESHOLD
# ============================================================================

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


# ============================================================================
# TEMPO DE INFERÊNCIA
# ============================================================================

def measure_inference_time(model, processor, records, cn_image_sets, ad_image_sets, id_ad, id_cn, max_slices,
                            n_warmup=N_INFERENCE_WARMUP, n_measure=N_INFERENCE_MEASURE):
    sample = records[: n_warmup + n_measure]
    for r in sample[:n_warmup]:
        evaluate_patient(model, processor, r["patient_dir"], cn_image_sets, ad_image_sets, id_ad, id_cn, max_slices=max_slices)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times, n_slices_list = [], []
    for r in sample[n_warmup:]:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, n_slices = evaluate_patient(model, processor, r["patient_dir"], cn_image_sets, ad_image_sets,
                                        id_ad, id_cn, max_slices=max_slices)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        n_slices_list.append(n_slices)

    times = np.array(times)
    return {
        "device": str(DEVICE), "model_id": MODEL_ID, "mode": f"few-shot (k={K_SHOTS})",
        "n_patients_measured": int(len(times)),
        "mean_slices_per_patient": float(np.mean(n_slices_list)) if n_slices_list else 0.0,
        "mean_seconds_per_patient": float(times.mean()) if len(times) else 0.0,
        "std_seconds_per_patient": float(times.std()) if len(times) else 0.0,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="avaliar só os N primeiros pacientes do teste")
    parser.add_argument("--debug", action="store_true", help="imprime logits/tokens para os 3 primeiros pacientes do teste")
    parser.add_argument("--k-shots", type=int, default=K_SHOTS, help="exemplos (pacientes) por classe (padrão: 3)")
    parser.add_argument("--slices-per-exemplar", type=int, default=N_SLICES_PER_EXEMPLAR,
                         help="slices por paciente-exemplo (padrão: 3)")
    parser.add_argument("--max-slices", type=int, default=MAX_SLICES_PER_PATIENT)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-inference-timing", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)

    all_train_records = load_records(TRAIN_LABELS, TRAIN_DIR, "train")
    test_records = load_records(TEST_LABELS, TEST_DIR, "test")
    if args.limit:
        test_records = test_records[: args.limit]

    train_labels = [r["label"] for r in all_train_records]
    train_fit, val_internal = train_test_split(
        all_train_records, test_size=VAL_FRACTION, stratify=train_labels, random_state=SEED
    )
    print(f"\nSplit (idêntico aos outros scripts): {len(train_fit)} train_fit | "
          f"{len(val_internal)} val_internal (calibração) | {len(test_records)} test")

    cn_exemplar_records, ad_exemplar_records = select_exemplars(train_fit, k_shots=args.k_shots, seed=SEED)
    cn_ids = [r["patient_id"] for r in cn_exemplar_records]
    ad_ids = [r["patient_id"] for r in ad_exemplar_records]
    print(f"\nExemplos CN (few-shot, k={args.k_shots}): {cn_ids}")
    print(f"Exemplos AD (few-shot, k={args.k_shots}): {ad_ids}")
    with open(RESULTS_DIR / "exemplars_used.json", "w") as f:
        json.dump({"k_shots": args.k_shots, "cn_exemplars": cn_ids, "ad_exemplars": ad_ids}, f, indent=2)

    model, processor = load_model_and_processor(quantize_4bit=not args.no_4bit)
    id_ad, id_cn = get_answer_token_ids(processor)

    cn_image_sets = [representative_slice_images(r, n_slices=args.slices_per_exemplar, max_slices=args.max_slices)
                      for r in cn_exemplar_records]
    ad_image_sets = [representative_slice_images(r, n_slices=args.slices_per_exemplar, max_slices=args.max_slices)
                      for r in ad_exemplar_records]
    n_exemplar_images = (len(cn_image_sets) + len(ad_image_sets)) * args.slices_per_exemplar
    print(f"\nTotal de imagens de exemplo no prompt: {n_exemplar_images} "
          f"({len(cn_image_sets)} pacientes CN + {len(ad_image_sets)} pacientes AD, "
          f"{args.slices_per_exemplar} slices cada) + 1 imagem de consulta por chamada")

    print(f"\n===== Calibrando threshold em val_internal (few-shot, k={args.k_shots}) =====")
    val_df = evaluate_population(model, processor, val_internal, cn_image_sets, ad_image_sets, id_ad, id_cn,
                                  args.max_slices, desc="Validação interna")
    threshold, score = find_best_threshold(val_df["y_true"].values, val_df["probability"].values)
    print(f"Threshold calibrado: {threshold:.2f} ({SELECTION_METRIC} na validação = {score:.4f})")

    print(f"\n===== Avaliando test/ (few-shot, k={args.k_shots}, uma única vez) =====")
    test_df = evaluate_population(model, processor, test_records, cn_image_sets, ad_image_sets, id_ad, id_cn,
                                   args.max_slices, verbose_first_n=(3 if args.debug else 0), desc="Teste")

    y_true, y_prob = test_df["y_true"].values, test_df["probability"].values
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"\n===== RESULTADOS FINAIS (test/, nível paciente, few-shot k={args.k_shots}) =====")
    for k in ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]:
        print(f"  {k:<20}: {metrics[k]:.4f}")

    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump({"threshold": threshold, "k_shots": args.k_shots, "n_patients_test": len(y_true),
                    "metrics": metrics, "cn_exemplars": cn_ids, "ad_exemplars": ad_ids}, f, indent=4)

    report = classification_report(y_true, y_pred, target_names=[CLASS_1, CLASS_2], digits=4)
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Threshold: {threshold:.2f}\nK-shots: {args.k_shots}\nN: {len(y_true)}\n\n{report}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[CLASS_1, CLASS_2], yticklabels=[CLASS_1, CLASS_2])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — MedGemma few-shot k={args.k_shots} (n={len(y_true)})\n"
              f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  Threshold={threshold:.2f}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    test_df["y_pred"] = y_pred
    test_df.to_csv(RESULTS_DIR / "predictions_test.csv", index=False)

    if not args.skip_inference_timing and len(test_records) > N_INFERENCE_WARMUP:
        print("\n===== Medindo tempo de inferência (few-shot) =====")
        inference_stats = measure_inference_time(model, processor, test_records, cn_image_sets, ad_image_sets,
                                                   id_ad, id_cn, args.max_slices)
        print(json.dumps(inference_stats, indent=2))
        with open(RESULTS_DIR / "inference_time.json", "w") as f:
            json.dump(inference_stats, f, indent=4)

    print(f"\nTudo salvo em: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()