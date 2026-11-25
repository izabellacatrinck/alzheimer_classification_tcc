"""
Avalia o MedGemma-4B-IT em modo ZERO-SHOT (contexto de pesquisa + pergunta +
imagem; nenhum exemplo) para classificação binária CN vs AD, com o MESMO
protocolo do one-shot v3: mesmo split train/test (sem cross-validation), mesma
validação interna (VAL_FRACTION, SEED), mesma extração do score, mesma busca de
limiar e avaliação final única no test/ a cada rodada.

EXTRAÇÃO DO SCORE (igual à do one-shot v3, em TODOS os modos, inclusive o full):
    d = logsumexp(logp[variantes de Yes]) - logsumexp(logp[variantes de No])
  no primeiro token gerado (decodificação gulosa, max_new_tokens=1). As variantes
  cobrem "Yes"/" Yes"/"yes"/... e "No"/" No"/"no"/...: o modelo gera "No" sem
  espaço, e os tokens com espaço usados antes tinham massa ~1e-8. A massa total
  na família Yes/No é registrada por fatia e o script avisa se a mediana < 0,5.
  Score do paciente = MÉDIA de d entre as fatias. probability = sigmoid(score)
  é só para exibição. O limiar é buscado no domínio do score (pontos médios entre
  scores únicos da validação interna, mais um valor abaixo do mínimo e um acima
  do máximo; em empate, o candidato mais próximo da mediana dos scores).
  O AUC sai com IC95% por bootstrap estratificado.

VERIFICAÇÃO DOS DADOS: antes de rodar, o script confere que todo paciente tem
exatamente MAX_SLICES_PER_PATIENT (9) fatias em disco e aborta se não tiver
(use --allow-extra-slices só se isso for intencional).

PERGUNTAS. As fatias do dataset são axiais; o hipocampo quase não aparece, mas
ventrículos aumentados e sulcos alargados aparecem. O prompt "original" ancora o
modelo em hipocampo/lobo temporal medial; as variantes (visible_features,
ventricles, sulci) perguntam sobre atrofia visível e medem atrofia, não
diagnóstico. Use --mode compare-prompts (só val_internal, nunca o test/) para
comparar e só então rode --mode full --prompt <escolhido>. A pergunta é a mesma
do one-shot v3, sem a referência aos exemplos.

Modos:
    # 1) comparar prompts em val_internal (não toca no test/):
    python evaluate_zero_shot.py --mode compare-prompts
    python evaluate_zero_shot.py --mode compare-prompts --prompts original ventricles --limit-val 30

    # 2) descrição livre em val_internal:
    python evaluate_zero_shot.py --mode describe --describe-n 4

    # 3) rodada completa (calibra em val_internal, avalia test/ uma única vez):
    python evaluate_zero_shot.py --mode full --prompt original --limit 3 --debug
    python evaluate_zero_shot.py --mode full --prompt original

Resultados em results_medgemma_zeroshot_CN_vs_AD/v2/ (o modo full grava em
full_<prompt>_families/, sem sobrescrever os resultados da extração antiga).

Requisitos:
    pip install transformers accelerate bitsandbytes torch pillow pandas numpy tqdm scikit-learn matplotlib seaborn
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
# CONFIGURAÇÃO — idêntica ao train_densenet_single.py / one-shot / few-shot
# ============================================================================

SEED = 42
MODEL_ID = "google/medgemma-4b-it"

CLASS_1 = "CN"  # label 0
CLASS_2 = "AD"  # label 1

TRAIN_DIR = Path("Data/ADNI_FINAL_Updated/train")
TEST_DIR = Path("Data/ADNI_FINAL_Updated/test")
TRAIN_LABELS = Path("Data/SPLITS_U/train_labels.csv")
TEST_LABELS = Path("Data/SPLITS_U/test_labels.csv")

RESULTS_DIR = Path(f"results_medgemma_zeroshot_{CLASS_1}_vs_{CLASS_2}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
V2_DIR = RESULTS_DIR / "v2"  # resultados desta versão (não sobrescreve os antigos)

VAL_FRACTION = 0.15
MAX_SLICES_PER_PATIENT = 9  # o dataset tem 9 fatias por paciente (conferido em check_slice_counts)
SELECTION_METRIC = "balanced_accuracy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

N_INFERENCE_WARMUP = 2
N_INFERENCE_MEASURE = 10
N_BOOT = 2000  # reamostragens do bootstrap para o IC95% da AUC

# Mantidos só para compatibilidade com diagnose_zero_shot.py (extração ANTIGA,
# com espaço). A extração usada nesta versão são as famílias abaixo.
ANSWER_TOKEN_AD = " Yes"
ANSWER_TOKEN_CN = " No"

YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO", " NO"]

# Idêntico, palavra por palavra, ao RESEARCH_CONTEXT do one-shot v3.
RESEARCH_CONTEXT = (
    "This is part of an academic research study on automated Alzheimer's "
    "disease classification using the de-identified ADNI neuroimaging "
    "dataset. You are acting as an expert neuroradiologist assisting "
    "with a structured research classification task — this is a "
    "scientific evaluation of model performance on a research "
    "benchmark, not a clinical diagnosis for an individual patient."
)
# Mesma pergunta do one-shot v3 (PROMPTS["original"]), sem a referência aos
# exemplos, que não existem neste experimento.
QUERY_PROMPT = (
    "Now consider the following patient slice. Established structural "
    "markers of Alzheimer's disease on MRI include hippocampal atrophy, "
    "ventricular enlargement, and cortical thinning, particularly in the "
    "medial temporal lobe. Considering these features, does this slice "
    "show signs of Alzheimer's disease? Answer with only Yes or No."
)

# [2] Perguntas sobre o que é de fato visível nas fatias axiais do dataset.
# Todas usam o mesmo RESEARCH_CONTEXT; só muda a pergunta.
PROMPTS = {
    "original": QUERY_PROMPT,
    "visible_features": (
        "Now consider the following axial brain MRI slice. Compared with a "
        "cognitively normal adult of similar age, are the lateral ventricles "
        "enlarged and the cortical sulci widened, indicating brain atrophy? "
        "Answer with only Yes or No."
    ),
    "ventricles": (
        "Now consider the following axial brain MRI slice. Are the lateral "
        "ventricles enlarged compared with a cognitively normal adult of "
        "similar age? Answer with only Yes or No."
    ),
    "sulci": (
        "Now consider the following axial brain MRI slice. Are the cortical "
        "sulci widened compared with a cognitively normal adult of similar "
        "age? Answer with only Yes or No."
    ),
}

# [3] Descrição livre: tamanho dos ventrículos, sulcos, hipocampo, plano.
DESCRIBE_PROMPT = (
    "Describe this brain MRI slice in detail. State the imaging modality and "
    "the anatomical plane, whether the lateral ventricles look normal or "
    "enlarged, whether the cortical sulci look normal or widened, and whether "
    "the hippocampus or temporal horns are visible in this slice."
)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# CARREGAMENTO DE IMAGENS / REGISTROS — idêntico aos outros scripts
# ============================================================================

def select_slices(patient_dir, max_slices=MAX_SLICES_PER_PATIENT):
    npy_files = sorted(Path(patient_dir).glob("*.npy"))
    n = len(npy_files)
    if n <= max_slices:
        return npy_files
    start = (n - max_slices) // 2
    return npy_files[start: start + max_slices]


def check_slice_counts(records, expected=MAX_SLICES_PER_PATIENT, allow_extra=False):
    """Confere que todo paciente tem exatamente `expected` fatias em disco.
    Com mais fatias, select_slices usaria apenas uma janela central da lista
    ordenada por nome, e o experimento deixaria de usar as fatias selecionadas."""
    counts = {r["patient_id"]: len(list(Path(r["patient_dir"]).glob("*.npy"))) for r in records}
    bad = {pid: n for pid, n in counts.items() if n != expected}
    if not bad:
        print(f"Verificação: os {len(counts)} pacientes têm exatamente {expected} fatias.")
        return
    dist = pd.Series(list(bad.values())).value_counts().sort_index().to_dict()
    print(f"\n*** {len(bad)} de {len(counts)} pacientes NÃO têm {expected} fatias "
          f"(contagem de arquivos -> nº de pacientes: {dist}). ***")
    if not allow_extra:
        raise RuntimeError(
            f"Esperadas {expected} fatias por paciente. Confira a pasta de dados antes de rodar "
            f"(ou use --allow-extra-slices, ciente de que só a janela central de {expected} "
            f"arquivos, em ordem alfabética, será usada)."
        )


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
    """EXTRAÇÃO ANTIGA (tokens com espaço). Mantida só para o diagnose_zero_shot.py."""
    tok_ad = processor.tokenizer.encode(ANSWER_TOKEN_AD, add_special_tokens=False)
    tok_cn = processor.tokenizer.encode(ANSWER_TOKEN_CN, add_special_tokens=False)
    if len(tok_ad) != 1 or len(tok_cn) != 1:
        print(f"AVISO: tokens candidatos não são de 1 único token (AD={tok_ad}, CN={tok_cn}).")
    id_ad, id_cn = tok_ad[0], tok_cn[0]
    print(f"Confirmação: id {id_ad} -> {processor.tokenizer.decode([id_ad])!r} | "
          f"id {id_cn} -> {processor.tokenizer.decode([id_cn])!r}")
    return id_ad, id_cn


def get_answer_token_families(processor):
    """[4] Famílias de tokens Yes/No (com e sem espaço, maiúscula/minúscula)."""
    tok = processor.tokenizer

    def collect(variants):
        mapping = {}
        for v in variants:
            ids = tok.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                mapping[v] = ids[0]
        return mapping

    yes_map, no_map = collect(YES_VARIANTS), collect(NO_VARIANTS)
    if not yes_map or not no_map:
        raise RuntimeError("Nenhuma variante de Yes/No virou 1 único token — revise YES_VARIANTS/NO_VARIANTS.")
    print("Tokens Yes:", {k: (i, tok.convert_ids_to_tokens(i)) for k, i in yes_map.items()})
    print("Tokens No :", {k: (i, tok.convert_ids_to_tokens(i)) for k, i in no_map.items()})
    return {"yes": sorted(set(yes_map.values())), "no": sorted(set(no_map.values()))}


# ============================================================================
# SCORING ZERO-SHOT (sem exemplos — só RESEARCH_CONTEXT + pergunta + imagem)
# ============================================================================

def build_messages(prompt_name, image):
    return [
        {"role": "user", "content": [
            {"type": "text", "text": RESEARCH_CONTEXT},
            {"type": "text", "text": PROMPTS[prompt_name]},
            {"type": "image", "image": image},
        ]}
    ]


@torch.inference_mode()
def score_slice(model, processor, image, tokens, prompt_name="original", verbose=False):
    """Retorna d = log P(família Yes) - log P(família No) para o 1º token gerado,
    a massa de probabilidade total da família Yes/No e o token mais provável."""
    inputs = processor.apply_chat_template(
        build_messages(prompt_name, image), add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=DTYPE)

    outputs = model.generate(
        **inputs, max_new_tokens=1, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )
    logp = torch.log_softmax(outputs.scores[0][0].float(), dim=-1)
    log_yes = torch.logsumexp(logp[tokens["yes"]], dim=0)
    log_no = torch.logsumexp(logp[tokens["no"]], dim=0)
    d = (log_yes - log_no).item()
    mass = torch.exp(torch.logaddexp(log_yes, log_no)).item()
    top1 = processor.tokenizer.decode([int(torch.argmax(logp).item())])

    if verbose:
        print(f"      [debug] d=logP(Yes)-logP(No)={d:+.3f}  massa Yes/No={mass:.4f}  "
              f"token mais provável: {top1!r}")

    return {"d": d, "mass": mass, "top1": top1}


def evaluate_patient(model, processor, patient_dir, tokens, prompt_name, max_slices, verbose=False):
    rows = []
    for si, p in enumerate(select_slices(patient_dir, max_slices=max_slices)):
        r = score_slice(model, processor, load_slice_as_image(p), tokens, prompt_name, verbose=verbose)
        r["slice_idx"] = si
        rows.append(r)
    return rows


def evaluate_population(model, processor, records, tokens, prompt_name, max_slices,
                        verbose_first_n=0, desc="Avaliando"):
    """Retorna (patient_df, slice_df). score do paciente = média de d entre as fatias."""
    label_map = {CLASS_1: 0, CLASS_2: 1}
    patient_rows, slice_rows = [], []
    for i, record in enumerate(tqdm(records, desc=desc)):
        verbose = i < verbose_first_n
        rows = evaluate_patient(model, processor, record["patient_dir"], tokens, prompt_name,
                                max_slices, verbose=verbose)
        score = float(np.mean([r["d"] for r in rows]))
        y = label_map[record["label"]]
        patient_rows.append({
            "patient_id": record["patient_id"], "y_true": y, "score": score,
            "probability": float(sigmoid(score)), "n_slices": len(rows),
            "mass_family": float(np.median([r["mass"] for r in rows])),
        })
        for r in rows:
            slice_rows.append({"patient_id": record["patient_id"], "y_true": y, **r})
        if verbose:
            print(f"  {record['patient_id']} (label={record['label']}) -> score={score:+.3f} "
                  f"(P={sigmoid(score):.4f})")
    return pd.DataFrame(patient_rows), pd.DataFrame(slice_rows)


def check_extraction(slice_df, name=""):
    """Avisa se o primeiro token gerado quase nunca é Yes/No (extração inválida)."""
    med_mass = slice_df["mass"].median()
    top = slice_df["top1"].value_counts().head(5).to_dict()
    print(f"  [extração{' ' + name if name else ''}] massa mediana na família Yes/No = {med_mass:.3f} | "
          f"top-1 mais frequentes: {top}")
    if med_mass < 0.5:
        print("  *** ALERTA: a maior parte da probabilidade do 1º token NÃO está em Yes/No. "
              "O score não é confiável — revise o prompt/tokens antes de interpretar métricas. ***")
    return med_mass


# ============================================================================
# MÉTRICAS / THRESHOLD
# ============================================================================

def find_best_threshold(y_true, scores, metric=SELECTION_METRIC):
    """Busca o limiar no domínio do score (diferença de log-probabilidades):
    candidatos são os pontos médios entre scores únicos (mais um valor abaixo do
    mínimo e um acima do máximo). Em empate, escolhe o candidato mais próximo da
    mediana dos scores, para evitar o caso degenerado 'tudo numa classe só'."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    uniq = np.sort(np.unique(scores))
    if len(uniq) == 1:
        candidates = np.array([uniq[0] - 1.0, uniq[0] + 1.0])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        candidates = np.concatenate([[uniq[0] - 1.0], mids, [uniq[-1] + 1.0]])

    best_score, best_ts = -1.0, []
    for t in candidates:
        preds = (scores >= t).astype(int)
        s = (balanced_accuracy_score(y_true, preds) if metric == "balanced_accuracy"
             else f1_score(y_true, preds, zero_division=0))
        if s > best_score + 1e-12:
            best_score, best_ts = s, [t]
        elif abs(s - best_score) <= 1e-12:
            best_ts.append(t)

    median_score = float(np.median(scores))
    best_t = min(best_ts, key=lambda t: abs(t - median_score))
    return float(best_t), float(best_score)


def auc_ci(y_true, scores, n_boot=N_BOOT, seed=SEED):
    """IC95% da AUC por bootstrap estratificado (reamostra pacientes por classe)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y_true[b], scores[b]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def compute_metrics(y_true, y_pred, scores):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["auc"] = float("nan")
    lo, hi = auc_ci(y_true, scores)
    metrics["auc_ci95_low"], metrics["auc_ci95_high"] = lo, hi
    return metrics


# ============================================================================
# TEMPO DE INFERÊNCIA
# ============================================================================

def measure_inference_time(model, processor, records, tokens, prompt_name, max_slices,
                           n_warmup=N_INFERENCE_WARMUP, n_measure=N_INFERENCE_MEASURE):
    sample = records[: n_warmup + n_measure]
    for r in sample[:n_warmup]:
        evaluate_patient(model, processor, r["patient_dir"], tokens, prompt_name, max_slices)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times, n_slices_list = [], []
    for r in sample[n_warmup:]:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rows = evaluate_patient(model, processor, r["patient_dir"], tokens, prompt_name, max_slices)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        n_slices_list.append(len(rows))

    times = np.array(times)
    return {
        "device": str(DEVICE), "model_id": MODEL_ID,
        "mode": "zero-shot (famílias de tokens Yes/No)", "prompt": prompt_name,
        "n_warmup_patients": int(n_warmup),
        "n_patients_measured": int(len(times)),
        "mean_slices_per_patient": float(np.mean(n_slices_list)) if n_slices_list else 0.0,
        "mean_seconds_per_patient": float(times.mean()) if len(times) else 0.0,
        "std_seconds_per_patient": float(times.std()) if len(times) else 0.0,
    }


# ============================================================================
# [2] MODO compare-prompts — só val_internal, nunca toca no test/
# ============================================================================

def summarize_prompt(name, patient_df, slice_df):
    y, s = patient_df["y_true"].values, patient_df["score"].values
    p_auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
    lo, hi = auc_ci(y, s)
    sl_auc = (roc_auc_score(slice_df["y_true"], slice_df["d"])
              if slice_df["y_true"].nunique() == 2 else float("nan"))
    print(f"\n--- prompt: {name} ---")
    print(f"  AUC paciente = {p_auc:.3f}  IC95% = [{lo:.3f}, {hi:.3f}]  |  AUC fatia = {sl_auc:.3f}")
    print(f"  d: média={slice_df['d'].mean():+.2f}  std={slice_df['d'].std():.2f}  "
          f"fração(Yes>No)={(slice_df['d'] > 0).mean():.0%}")
    check_extraction(slice_df, name)
    return {"prompt": name, "auc_patient": p_auc, "auc_ci95_low": lo, "auc_ci95_high": hi,
            "auc_slice": sl_auc, "d_mean": float(slice_df["d"].mean()),
            "frac_yes": float((slice_df["d"] > 0).mean()),
            "mass_median": float(slice_df["mass"].median())}


def run_compare_prompts(args, model, processor, tokens, val_records):
    out_dir = V2_DIR / "compare_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.prompts or list(PROMPTS)

    print(f"\n===== Comparando prompts em val_internal ({len(val_records)} pacientes): {names} =====")
    summaries, all_patients, all_slices = [], [], []
    for name in names:
        p_df, s_df = evaluate_population(model, processor, val_records, tokens, name,
                                         args.max_slices, desc=f"val/{name}")
        p_df["prompt"], s_df["prompt"] = name, name
        all_patients.append(p_df)
        all_slices.append(s_df)
        summaries.append(summarize_prompt(name, p_df, s_df))

    pd.concat(all_patients).to_csv(out_dir / "val_patient_scores.csv", index=False)
    pd.concat(all_slices).to_csv(out_dir / "val_slice_scores.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "val_prompt_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("RESUMO (AUC por paciente em val_internal, IC95% bootstrap)")
    print("=" * 78)
    for s in sorted(summaries, key=lambda x: -x["auc_patient"]):
        print(f"  {s['prompt']:<18} AUC={s['auc_patient']:.3f} "
              f"[{s['auc_ci95_low']:.3f}, {s['auc_ci95_high']:.3f}]  "
              f"massa Yes/No={s['mass_median']:.2f}  fração Yes={s['frac_yes']:.0%}")
    print("\nCuidado: escolher o melhor entre vários prompts pelo AUC de val_internal tem viés "
          "otimista. Só vale trocar o prompt se o IC de um deles ficar claramente acima de 0,5; "
          "caso contrário, o resultado honesto é 'sem sinal em nenhum'. O test/ não foi tocado.")
    print(f"Salvo em {out_dir}/")


# ============================================================================
# [3] MODO describe — descrição livre em val_internal
# ============================================================================

@torch.inference_mode()
def describe_slice(model, processor, image, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": DESCRIBE_PROMPT}, {"type": "image", "image": image}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=DTYPE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def run_describe(args, model, processor, val_records):
    out_dir = V2_DIR / "describe"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"\n===== Descrição livre ({args.describe_tokens} tokens) em val_internal =====")
    blocks = []
    for cls in (CLASS_1, CLASS_2):
        pool = [r for r in val_records if r["label"] == cls]
        for rec in rng.sample(pool, min(args.describe_n, len(pool))):
            paths = select_slices(rec["patient_dir"], args.max_slices)
            image = load_slice_as_image(paths[len(paths) // 2])  # fatia do meio
            text = describe_slice(model, processor, image, args.describe_tokens)
            block = f"[{cls} {rec['patient_id']}]\n{text}\n"
            print(block)
            blocks.append(block)

    (out_dir / "describe.txt").write_text("\n".join(blocks))
    print(f"Salvo em {out_dir / 'describe.txt'}")


# ============================================================================
# MODO full — calibra em val_internal, avalia test/ uma única vez
# ============================================================================

def run_full(args, model, processor, val_internal, test_records):
    prompt_name = args.prompt
    out_dir = V2_DIR / f"full_{prompt_name}_families"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = get_answer_token_families(processor)

    print(f"\n===== Calibrando limiar em val_internal (zero-shot, prompt='{prompt_name}') =====")
    val_df, val_slices = evaluate_population(
        model, processor, val_internal, tokens, prompt_name, args.max_slices,
        desc="Validação interna",
    )
    val_mass = check_extraction(val_slices, "val")
    val_df.to_csv(out_dir / "val_patient_scores.csv", index=False)
    val_slices.to_csv(out_dir / "val_slice_scores.csv", index=False)

    threshold, score = find_best_threshold(val_df["y_true"].values, val_df["score"].values)
    print(f"Limiar calibrado (score = diferença de log-probabilidades): {threshold:+.3f} "
          f"[sigmoid={sigmoid(threshold):.4f}] ({SELECTION_METRIC} na validação = {score:.4f})")

    print("\n===== Avaliando test/ (zero-shot, uma única vez nesta rodada) =====")
    test_df, test_slices = evaluate_population(
        model, processor, test_records, tokens, prompt_name, args.max_slices,
        verbose_first_n=(3 if args.debug else 0), desc="Teste",
    )
    test_mass = check_extraction(test_slices, "test")
    test_slices.to_csv(out_dir / "slice_scores_test.csv", index=False)

    y_true, scores = test_df["y_true"].values, test_df["score"].values
    y_pred = (scores >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, scores)

    print("\n===== RESULTADOS FINAIS (test/, nível paciente, zero-shot) =====")
    for k in ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]:
        print(f"  {k:<20}: {metrics[k]:.4f}")
    print(f"  {'auc IC95%':<20}: [{metrics['auc_ci95_low']:.4f}, {metrics['auc_ci95_high']:.4f}]")

    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump({"prompt": prompt_name, "threshold_score": float(threshold),
                   "threshold_sigmoid": float(sigmoid(threshold)),
                   "n_patients_test": int(len(y_true)),
                   "val_median_mass_yes_no": float(val_mass),
                   "test_median_mass_yes_no": float(test_mass),
                   "yes_token_ids": tokens["yes"], "no_token_ids": tokens["no"],
                   "metrics": metrics}, f, indent=4)

    report = classification_report(y_true, y_pred, target_names=[CLASS_1, CLASS_2], digits=4)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Prompt: {prompt_name}\nLimiar (score): {threshold:+.3f}\nN: {len(y_true)}\n\n{report}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[CLASS_1, CLASS_2], yticklabels=[CLASS_1, CLASS_2])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — MedGemma zero-shot [{prompt_name}] (n={len(y_true)})\n"
              f"AUC={metrics['auc']:.3f}  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"Threshold(score)={threshold:+.2f}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    test_df["y_pred"] = y_pred
    test_df.to_csv(out_dir / "predictions_test.csv", index=False)

    if not args.skip_inference_timing and len(test_records) > N_INFERENCE_WARMUP:
        print("\n===== Medindo tempo de inferência (zero-shot) =====")
        inference_stats = measure_inference_time(model, processor, test_records, tokens, prompt_name,
                                                 args.max_slices)
        print(json.dumps(inference_stats, indent=2))
        with open(out_dir / "inference_time.json", "w") as f:
            json.dump(inference_stats, f, indent=4)

    print(f"\nTudo salvo em: {out_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["full", "compare-prompts", "describe"], default="full")
    parser.add_argument("--prompt", choices=list(PROMPTS), default="original",
                        help="[full] pergunta usada na calibração e no teste")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=None,
                        help="[compare-prompts] subconjunto de prompts (padrão: todos)")
    parser.add_argument("--limit", type=int, default=None, help="[full] avaliar só os N primeiros pacientes do teste")
    parser.add_argument("--limit-val", type=int, default=None, help="usar só N pacientes de val_internal")
    parser.add_argument("--debug", action="store_true",
                        help="[full] imprime logits/probabilidade/token para os 3 primeiros pacientes do teste")
    parser.add_argument("--describe-n", type=int, default=3, help="[describe] pacientes por classe")
    parser.add_argument("--describe-tokens", type=int, default=250, help="[describe] max_new_tokens")
    parser.add_argument("--max-slices", type=int, default=MAX_SLICES_PER_PATIENT)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-inference-timing", action="store_true")
    parser.add_argument("--allow-extra-slices", action="store_true",
                        help="não abortar se algum paciente não tiver exatamente 9 fatias em disco")
    args = parser.parse_args()

    seed_everything(SEED)
    V2_DIR.mkdir(parents=True, exist_ok=True)

    all_train_records = load_records(TRAIN_LABELS, TRAIN_DIR, "train")
    train_labels = [r["label"] for r in all_train_records]
    train_fit, val_internal = train_test_split(
        all_train_records, test_size=VAL_FRACTION, stratify=train_labels, random_state=SEED
    )
    if args.limit_val:
        val_internal = val_internal[: args.limit_val]

    test_records = []
    if args.mode == "full":
        test_records = load_records(TEST_LABELS, TEST_DIR, "test")
        if args.limit:
            test_records = test_records[: args.limit]

    check_slice_counts(all_train_records + test_records, MAX_SLICES_PER_PATIENT, args.allow_extra_slices)

    print(f"\nSplit (idêntico aos outros scripts): {len(train_fit)} train_fit (não usado — "
          f"zero-shot não tem exemplo) | {len(val_internal)} val_internal | "
          f"{len(test_records)} test (só no modo full)")

    model, processor = load_model_and_processor(quantize_4bit=not args.no_4bit)
    if args.mode == "compare-prompts":
        tokens = get_answer_token_families(processor)
        run_compare_prompts(args, model, processor, tokens, val_internal)
    elif args.mode == "describe":
        run_describe(args, model, processor, val_internal)
    else:
        run_full(args, model, processor, val_internal, test_records)


if __name__ == "__main__":
    main()