"""
Avalia o MedGemma-4B-IT em modo FEW-SHOT com K pacientes-exemplo POR CLASSE
(K=5 por padrão: 5 pacientes CN + 5 pacientes AD = 10 exemplos) para
classificação binária CN vs AD — mesmo protocolo do zero-shot e do one-shot:
mesmo split (SEED, VAL_FRACTION), threshold calibrado em val_internal,
avaliação final única no test/.

ESTE SCRIPT É AUTOCONTIDO: não importa nenhum outro script do projeto. A
lógica compartilhada com o zero-shot/one-shot (extração por famílias de
tokens Yes/No, agregação por paciente, busca de threshold, AUC com IC95%,
prompts, split) foi COPIADA literalmente para cá. Se você alterar algum
desses trechos no zero-shot ou no one-shot, altere aqui também, senão os
experimentos deixam de ser comparáveis. Trechos copiados estão marcados com
"[CÓPIA do zero-shot]".

DECISÕES DE DESENHO:

  - 9 FATIAS POR PACIENTE-EXEMPLO (--slices-per-exemplar 9, padrão), usando
    até 9 imagens também para cada paciente avaliado. Com K=3 isso dá até
    54 imagens de exemplo + 1 de consulta por chamada; acompanhe a memória.
    Reduza --slices-per-exemplar ou --k-shots se faltar memória.
  - O 1º exemplo de cada classe é o MESMO par do one-shot (a escolha usa a
    mesma regra e a mesma seed: random.Random(seed).choice em train_fit).
    Os K-1 restantes por classe são sorteados de train_fit. Com K=1 há um
    paciente-exemplo por classe, mas o contexto usa 9 fatias por paciente.
  - Pergunta: a do zero-shot (--prompt) acrescida de UMA referência aos
    exemplos — "Comparing this slice to the <2K por extenso> examples above,
    and" no prompt 'original', ou a frase "Use the <2K por extenso> examples
    above as a reference." nos demais. Asserts no import garantem que
    removendo a referência volta-se exatamente ao texto do zero-shot.
  - Os exemplos alternam as classes (CN, AD, CN, AD, ...), como em Ferber et
    al. (k=1, 3, 5 e 10 em classificação binária de imagens médicas). Como a
    ÚLTIMA imagem de exemplo é sempre da mesma classe, há risco de viés de
    recência: --order ad-first inverte a alternância (AD, CN, AD, CN, ...).
    Rode os dois; se o viés trocar de lado, é recência.
  - Use o mesmo --prompt escolhido em val_internal para o zero-shot. Não
    reescolha o prompt por experimento.

Proteções: os exemplos vêm só de train_fit; o script confere que são
pacientes distintos e que nenhum está em val_internal ou test/. O score é a
diferença de log-probabilidades entre as famílias de tokens Yes/No (com e
sem espaço), porque o modelo gera "No" sem espaço; o script avisa se a massa
na família Yes/No for baixa.

Resultados em results_medgemma_fewshot_CN_vs_AD/v2/full_<prompt>_k<K>_s<S>[...]/
(os resultados antigos não são sobrescritos).

Requisitos:
    pip install transformers accelerate bitsandbytes torch pillow pandas numpy tqdm scikit-learn matplotlib seaborn

Uso:
    # Teste rápido primeiro (confira memória e as massas Yes/No):
    python evaluate_medgemma_few_shot.py --prompt original --limit 3 --limit-val 6 --debug

    # Rodada completa (mesmo --prompt do zero-shot/one-shot):
    python evaluate_medgemma_few_shot.py --prompt <escolhido>

    # Controle de recência (ordem invertida) e robustez à escolha dos pacientes:
    python evaluate_medgemma_few_shot.py --prompt <escolhido> --order ad-first
    python evaluate_medgemma_few_shot.py --prompt <escolhido> --exemplar-seed 1
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
# CONFIGURAÇÃO — [CÓPIA do zero-shot] idêntica ao train_densenet_single.py
# ============================================================================

SEED = 42
MODEL_ID = "google/medgemma-4b-it"

CLASS_1 = "CN"  # label 0
CLASS_2 = "AD"  # label 1

TRAIN_DIR = Path("Data/ADNI_FINAL_Updated/train")
TEST_DIR = Path("Data/ADNI_FINAL_Updated/test")
TRAIN_LABELS = Path("Data/SPLITS_U/train_labels.csv")
TEST_LABELS = Path("Data/SPLITS_U/test_labels.csv")

RESULTS_DIR = Path(f"results_medgemma_fewshot_{CLASS_1}_vs_{CLASS_2}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
V2_DIR = RESULTS_DIR / "v2"  # resultados desta versão (não sobrescreve os antigos)
# Só para um aviso opcional (o arquivo é apenas LIDO, nenhum código do one-shot é importado):
ONE_SHOT_EXEMPLARS_FILE = Path(f"results_medgemma_oneshot_{CLASS_1}_vs_{CLASS_2}/exemplars_used.json")

VAL_FRACTION = 0.15
MAX_SLICES_PER_PATIENT = 9
SELECTION_METRIC = "balanced_accuracy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

N_BOOT = 2000  # reamostragens do bootstrap para o IC95% da AUC

# Few-shot
K_SHOTS = 3  # pacientes-exemplo POR CLASSE
N_SLICES_PER_EXEMPLAR = 9  # fatias por paciente-exemplo
IMAGES_WARNING_THRESHOLD = 20  # avisa se o prompt passar disso (memória da GPU)
N_INFERENCE_WARMUP = 2
N_INFERENCE_MEASURE = 5  # few-shot processa 2*K*S+1 imagens/slice — orçamento bem menor

# [CÓPIA do zero-shot] famílias de tokens para o score
YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO", " NO"]

# [CÓPIA do zero-shot] idêntico, palavra por palavra, ao RESEARCH_CONTEXT dos outros scripts.
RESEARCH_CONTEXT = (
    "This is part of an academic research study on automated Alzheimer's "
    "disease classification using the de-identified ADNI neuroimaging "
    "dataset. You are acting as an expert neuroradiologist assisting "
    "with a structured research classification task — this is a "
    "scientific evaluation of model performance on a research "
    "benchmark, not a clinical diagnosis for an individual patient."
)

# [CÓPIA do zero-shot] perguntas sem referência a exemplos.
PROMPTS = {
    "original": (
        "Now consider the following patient slice. Established structural "
        "markers of Alzheimer's disease on MRI include hippocampal atrophy, "
        "ventricular enlargement, and cortical thinning, particularly in the "
        "medial temporal lobe. Considering these features, does this slice "
        "show signs of Alzheimer's disease? Answer with only Yes or No."
    ),
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

# Legendas para exemplos com uma fatia.
CN_TEXT_1 = (
    "Example {n}: this brain MRI slice is from a patient with preserved "
    "hippocampal volume and no neurodegenerative abnormalities "
    "(Cognitively Normal)."
)
AD_TEXT_1 = (
    "Example {n}: this brain MRI slice is from a patient with hippocampal "
    "atrophy and enlarged ventricles consistent with Alzheimer's "
    "disease."
)
# Com mais de 1 fatia por paciente-exemplo (--slices-per-exemplar > 1).
CN_TEXT_MULTI = (
    "Example {n} ({n_slices} MRI slices from one patient): this patient "
    "has preserved hippocampal volume and no neurodegenerative "
    "abnormalities (Cognitively Normal)."
)
AD_TEXT_MULTI = (
    "Example {n} ({n_slices} MRI slices from one patient): this patient "
    "has hippocampal atrophy and enlarged ventricles consistent with "
    "Alzheimer's disease."
)

NUMBER_WORDS = {2: "two", 4: "four", 6: "six", 8: "eight", 10: "ten", 12: "twelve",
                14: "fourteen", 16: "sixteen", 18: "eighteen", 20: "twenty"}


def number_word(n):
    return NUMBER_WORDS.get(n, str(n))


def few_shot_query(prompt_name, n_examples):
    """Pergunta do zero-shot + UMA referência aos n_examples exemplos."""
    word = number_word(n_examples)
    base = PROMPTS[prompt_name]
    if prompt_name == "original":
        return base.replace(
            "Considering these features,",
            f"Comparing this slice to the {word} examples above, and considering these features,")
    first, rest = base.split(". ", 1)
    return f"{first}. Use the {word} examples above as a reference. {rest}"


def _check_prompt_consistency():
    """Removendo a referência aos exemplos, volta-se exatamente ao texto do zero-shot."""
    for name, base in PROMPTS.items():
        q = few_shot_query(name, 10)
        if name == "original":
            stripped = q.replace("Comparing this slice to the ten examples above, and considering",
                                 "Considering")
        else:
            stripped = q.replace(" Use the ten examples above as a reference.", "")
        assert stripped == base, f"O prompt '{name}' do few-shot difere do zero-shot além da referência aos exemplos."
        assert "the two examples above" in few_shot_query(name, 2)


_check_prompt_consistency()


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# CARREGAMENTO DE IMAGENS / REGISTROS — [CÓPIA do zero-shot]
# ============================================================================

def select_slices(patient_dir, max_slices=MAX_SLICES_PER_PATIENT):
    npy_files = sorted(Path(patient_dir).glob("*.npy"))
    n = len(npy_files)
    if n <= max_slices:
        return npy_files
    start = (n - max_slices) // 2
    return npy_files[start: start + max_slices]


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
# MODELO — [CÓPIA do zero-shot]
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


def get_answer_token_families(processor):
    """Famílias de tokens Yes/No (com e sem espaço, maiúscula/minúscula)."""
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
# MÉTRICAS / THRESHOLD — [CÓPIA do zero-shot]
# ============================================================================

def find_best_threshold(y_true, scores, metric=SELECTION_METRIC):
    """Busca o threshold no domínio do score (diferença de logits): candidatos
    são os pontos médios entre scores únicos (+ um abaixo do mínimo e um acima
    do máximo). Em empate, escolhe o candidato mais próximo da mediana dos
    scores, para evitar o caso degenerado 'tudo numa classe só'."""
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
# EXEMPLOS FEW-SHOT (K pacientes por classe; o nº1 é o mesmo par do one-shot)
# ============================================================================

def one_shot_pair(train_fit_records, seed=SEED):
    """MESMA regra do one-shot: um paciente CN e um AD sorteados de train_fit."""
    cn_candidates = [r for r in train_fit_records if r["label"] == CLASS_1]
    ad_candidates = [r for r in train_fit_records if r["label"] == CLASS_2]
    rng = random.Random(seed)
    return rng.choice(cn_candidates), rng.choice(ad_candidates)


def select_exemplars(train_fit_records, k_shots=K_SHOTS, seed=SEED):
    if k_shots < 1:
        raise ValueError("--k-shots precisa ser >= 1")
    cn_first, ad_first = one_shot_pair(train_fit_records, seed=seed)
    rng = random.Random(seed)
    cn_rest = [r for r in train_fit_records if r["label"] == CLASS_1 and r["patient_id"] != cn_first["patient_id"]]
    ad_rest = [r for r in train_fit_records if r["label"] == CLASS_2 and r["patient_id"] != ad_first["patient_id"]]
    if len(cn_rest) < k_shots - 1 or len(ad_rest) < k_shots - 1:
        raise ValueError(f"train_fit não tem pacientes suficientes para k={k_shots} por classe "
                         f"(CN extras: {len(cn_rest)}, AD extras: {len(ad_rest)}).")
    rng.shuffle(cn_rest)
    rng.shuffle(ad_rest)
    cn_selected = [cn_first] + cn_rest[: k_shots - 1]
    ad_selected = [ad_first] + ad_rest[: k_shots - 1]

    if seed == SEED and ONE_SHOT_EXEMPLARS_FILE.exists():  # só confere; nada é importado
        prev = json.loads(ONE_SHOT_EXEMPLARS_FILE.read_text())
        if (prev.get("cn_exemplar"), prev.get("ad_exemplar")) != (cn_first["patient_id"], ad_first["patient_id"]):
            print(f"AVISO: o par recalculado ({cn_first['patient_id']}, {ad_first['patient_id']}) difere do "
                  f"{ONE_SHOT_EXEMPLARS_FILE} ({prev.get('cn_exemplar')}, {prev.get('ad_exemplar')}). "
                  f"O split mudou desde o one-shot?")
    return cn_selected, ad_selected


def exemplar_images(record, n_slices=N_SLICES_PER_EXEMPLAR, max_slices=MAX_SLICES_PER_PATIENT):
    """n_slices imagens do paciente-exemplo. Com 1 fatia: a do meio, como no one-shot.
    Com mais: espaçadas igualmente entre as fatias selecionadas."""
    paths = select_slices(record["patient_dir"], max_slices=max_slices)
    if n_slices == 1:
        chosen = [paths[len(paths) // 2]]
    elif len(paths) <= n_slices:
        chosen = paths
    else:
        idx = sorted(set(np.linspace(0, len(paths) - 1, n_slices).round().astype(int)))
        chosen = [paths[i] for i in idx]
    return [load_slice_as_image(p) for p in chosen]


def build_exemplar_blocks(cn_image_sets, ad_image_sets, order="cn-first"):
    """Blocos de conteúdo (legenda + imagem(ns)) alternando as classes, como em
    Ferber et al.: CN, AD, CN, AD, ... (order='cn-first') ou AD, CN, AD, CN, ... ('ad-first')."""
    sequence = []
    for i in range(max(len(cn_image_sets), len(ad_image_sets))):
        pair = []
        if i < len(cn_image_sets):
            pair.append((CLASS_1, cn_image_sets[i]))
        if i < len(ad_image_sets):
            pair.append((CLASS_2, ad_image_sets[i]))
        if order == "ad-first":
            pair.reverse()
        sequence.extend(pair)

    blocks = []
    for n, (label, images) in enumerate(sequence, start=1):
        multi = len(images) > 1
        if label == CLASS_1:
            template = CN_TEXT_MULTI if multi else CN_TEXT_1
        else:
            template = AD_TEXT_MULTI if multi else AD_TEXT_1
        blocks.append({"type": "text", "text": template.format(n=n, n_slices=len(images))})
        for img in images:
            blocks.append({"type": "image", "image": img})
    return blocks


def build_messages(ctx, query_image):
    content = ([{"type": "text", "text": RESEARCH_CONTEXT}] + ctx["blocks"]
               + [{"type": "text", "text": ctx["query"]}, {"type": "image", "image": query_image}])
    return [{"role": "user", "content": content}]


# ============================================================================
# SCORING FEW-SHOT
# ============================================================================

@torch.inference_mode()
def score_slice_few_shot(model, processor, query_image, ctx, tokens, verbose=False):
    inputs = processor.apply_chat_template(
        build_messages(ctx, query_image), add_generation_prompt=True, tokenize=True,
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
              f"token mais provável: {top1!r}  (tokens no prompt: {inputs['input_ids'].shape[-1]})")

    return {"d": d, "mass": mass, "top1": top1}


def evaluate_patient(model, processor, patient_dir, ctx, tokens, max_slices, verbose=False):
    rows = []
    for si, p in enumerate(select_slices(patient_dir, max_slices=max_slices)):
        r = score_slice_few_shot(model, processor, load_slice_as_image(p), ctx, tokens, verbose=verbose)
        r["slice_idx"] = si
        rows.append(r)
    return rows


def evaluate_population(model, processor, records, ctx, tokens, max_slices,
                        verbose_first_n=0, desc="Avaliando"):
    """Retorna (patient_df, slice_df). score do paciente = média de d entre as fatias."""
    label_map = {CLASS_1: 0, CLASS_2: 1}
    patient_rows, slice_rows = [], []
    for i, record in enumerate(tqdm(records, desc=desc)):
        verbose = i < verbose_first_n
        rows = evaluate_patient(model, processor, record["patient_dir"], ctx, tokens, max_slices,
                                verbose=verbose)
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


# ============================================================================
# TEMPO DE INFERÊNCIA
# ============================================================================

def measure_inference_time(model, processor, records, ctx, tokens, max_slices, mode_label,
                           n_warmup=N_INFERENCE_WARMUP, n_measure=N_INFERENCE_MEASURE):
    sample = records[: n_warmup + n_measure]
    for r in sample[:n_warmup]:
        evaluate_patient(model, processor, r["patient_dir"], ctx, tokens, max_slices)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times, n_slices_list = [], []
    for r in sample[n_warmup:]:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rows = evaluate_patient(model, processor, r["patient_dir"], ctx, tokens, max_slices)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        n_slices_list.append(len(rows))

    times = np.array(times)
    return {
        "device": str(DEVICE), "model_id": MODEL_ID, "mode": mode_label,
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
    parser.add_argument("--prompt", choices=list(PROMPTS), default="original",
                        help="mesma pergunta do zero-shot/one-shot (escolha UM, em val_internal)")
    parser.add_argument("--k-shots", type=int, default=K_SHOTS, help="pacientes-exemplo POR CLASSE (padrão: 5)")
    parser.add_argument("--slices-per-exemplar", type=int, default=N_SLICES_PER_EXEMPLAR,
                        help="fatias por paciente-exemplo (padrão: 9)")
    parser.add_argument("--order", choices=["cn-first", "ad-first"], default="cn-first",
                        help="classe que abre cada par de exemplos alternados (controle de recência)")
    parser.add_argument("--exemplar-seed", type=int, default=SEED,
                        help="seed da escolha dos pacientes-exemplo (padrão: SEED, o par original do one-shot)")
    parser.add_argument("--limit", type=int, default=None, help="avaliar só os N primeiros pacientes do teste")
    parser.add_argument("--limit-val", type=int, default=None, help="usar só N pacientes de val_internal (teste rápido)")
    parser.add_argument("--debug", action="store_true",
                        help="imprime d/massa/token/nº de tokens para os 3 primeiros pacientes do teste")
    parser.add_argument("--max-slices", type=int, default=MAX_SLICES_PER_PATIENT)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-inference-timing", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)
    V2_DIR.mkdir(parents=True, exist_ok=True)

    all_train_records = load_records(TRAIN_LABELS, TRAIN_DIR, "train")
    test_records = load_records(TEST_LABELS, TEST_DIR, "test")
    if args.limit:
        test_records = test_records[: args.limit]

    train_labels = [r["label"] for r in all_train_records]
    train_fit, val_internal = train_test_split(
        all_train_records, test_size=VAL_FRACTION, stratify=train_labels, random_state=SEED
    )
    if args.limit_val:
        val_internal = val_internal[: args.limit_val]
    print(f"\nSplit (idêntico ao zero-shot/one-shot): {len(train_fit)} train_fit | "
          f"{len(val_internal)} val_internal (calibração) | {len(test_records)} test")

    cn_records, ad_records = select_exemplars(train_fit, k_shots=args.k_shots, seed=args.exemplar_seed)
    cn_ids = [r["patient_id"] for r in cn_records]
    ad_ids = [r["patient_id"] for r in ad_records]
    print(f"\nExemplos CN (few-shot, k={args.k_shots}): {cn_ids}")
    print(f"Exemplos AD (few-shot, k={args.k_shots}): {ad_ids}")

    # Proteção contra vazamento: exemplos só de train_fit, distintos entre si.
    other_ids = {r["patient_id"] for r in val_internal} | {r["patient_id"] for r in test_records}
    used = cn_ids + ad_ids
    assert len(set(used)) == len(used), "Paciente repetido entre os exemplos."
    assert not (set(used) & other_ids), "Paciente-exemplo também está em val_internal ou test/."

    suffix = ""
    if args.order != "cn-first":
        suffix += "_adfirst"
    if args.exemplar_seed != SEED:
        suffix += f"_ex{args.exemplar_seed}"
    out_dir = V2_DIR / f"full_{args.prompt}_k{args.k_shots}_s{args.slices_per_exemplar}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exemplars_used.json", "w") as f:
        json.dump({"k_shots": args.k_shots, "slices_per_exemplar": args.slices_per_exemplar,
                   "order": args.order, "exemplar_seed": args.exemplar_seed,
                   "cn_exemplars": cn_ids, "ad_exemplars": ad_ids}, f, indent=2)

    model, processor = load_model_and_processor(quantize_4bit=not args.no_4bit)
    tokens = get_answer_token_families(processor)

    cn_image_sets = [exemplar_images(r, n_slices=args.slices_per_exemplar, max_slices=args.max_slices)
                     for r in cn_records]
    ad_image_sets = [exemplar_images(r, n_slices=args.slices_per_exemplar, max_slices=args.max_slices)
                     for r in ad_records]
    n_examples = len(cn_records) + len(ad_records)
    n_exemplar_images = sum(len(s) for s in cn_image_sets) + sum(len(s) for s in ad_image_sets)
    ctx = {"blocks": build_exemplar_blocks(cn_image_sets, ad_image_sets, order=args.order),
           "query": few_shot_query(args.prompt, n_examples)}
    print(f"\nPrompt few-shot: {n_examples} exemplos ({len(cn_records)} CN + {len(ad_records)} AD), "
          f"{n_exemplar_images} imagens de exemplo + 1 de consulta por chamada "
          f"(~{(n_exemplar_images + 1) * 256} tokens de imagem), ordem '{args.order}'.")
    if n_exemplar_images + 1 > IMAGES_WARNING_THRESHOLD:
        print(f"  *** AVISO: mais de {IMAGES_WARNING_THRESHOLD} imagens por chamada — risco de falta de "
              f"memória e de viés de recência. Considere --k-shots menor ou --slices-per-exemplar 1. ***")
    print(f"Pergunta: {ctx['query']}")

    mode_label = (f"few-shot (k={args.k_shots}, s={args.slices_per_exemplar}, "
                  f"order={args.order}, prompt={args.prompt})")

    print(f"\n===== Calibrando threshold em val_internal ({mode_label}) =====")
    val_df, val_slices = evaluate_population(model, processor, val_internal, ctx, tokens, args.max_slices,
                                             desc="Validação interna")
    val_mass = check_extraction(val_slices, "val")
    val_df.to_csv(out_dir / "val_patient_scores.csv", index=False)
    val_slices.to_csv(out_dir / "val_slice_scores.csv", index=False)

    threshold, score = find_best_threshold(val_df["y_true"].values, val_df["score"].values)
    print(f"Threshold calibrado (score = diferença de logits): {threshold:+.3f} "
          f"[sigmoid={sigmoid(threshold):.4f}] ({SELECTION_METRIC} na validação = {score:.4f})")

    print(f"\n===== Avaliando test/ ({mode_label}, uma única vez) =====")
    test_df, test_slices = evaluate_population(model, processor, test_records, ctx, tokens, args.max_slices,
                                               verbose_first_n=(3 if args.debug else 0), desc="Teste")
    check_extraction(test_slices, "test")
    test_slices.to_csv(out_dir / "slice_scores_test.csv", index=False)

    y_true, scores = test_df["y_true"].values, test_df["score"].values
    y_pred = (scores >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, scores)

    print(f"\n===== RESULTADOS FINAIS (test/, nível paciente, {mode_label}) =====")
    for k in ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]:
        print(f"  {k:<20}: {metrics[k]:.4f}")
    print(f"  {'auc IC95%':<20}: [{metrics['auc_ci95_low']:.4f}, {metrics['auc_ci95_high']:.4f}]")

    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump({"prompt": args.prompt, "k_shots": args.k_shots,
                   "slices_per_exemplar": args.slices_per_exemplar, "order": args.order,
                   "exemplar_seed": args.exemplar_seed, "n_exemplar_images": n_exemplar_images,
                   "threshold_score": threshold, "threshold_sigmoid": float(sigmoid(threshold)),
                   "n_patients_test": len(y_true), "val_median_mass_yes_no": float(val_mass),
                   "yes_token_ids": tokens["yes"], "no_token_ids": tokens["no"],
                   "metrics": metrics, "cn_exemplars": cn_ids, "ad_exemplars": ad_ids}, f, indent=4)

    report = classification_report(y_true, y_pred, target_names=[CLASS_1, CLASS_2], digits=4)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Mode: {mode_label}\nThreshold (score): {threshold:+.3f}\nN: {len(y_true)}\n\n{report}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[CLASS_1, CLASS_2], yticklabels=[CLASS_1, CLASS_2])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — MedGemma few-shot k={args.k_shots} [{args.prompt}] (n={len(y_true)})\n"
              f"AUC={metrics['auc']:.3f}  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    test_df["y_pred"] = y_pred
    test_df.to_csv(out_dir / "predictions_test.csv", index=False)

    if not args.skip_inference_timing and len(test_records) > N_INFERENCE_WARMUP:
        print(f"\n===== Medindo tempo de inferência ({mode_label}) =====")
        inference_stats = measure_inference_time(model, processor, test_records, ctx, tokens,
                                                 args.max_slices, mode_label)
        print(json.dumps(inference_stats, indent=2))
        with open(out_dir / "inference_time.json", "w") as f:
            json.dump(inference_stats, f, indent=4)

    print(f"\nTudo salvo em: {out_dir}/")


if __name__ == "__main__":
    main()