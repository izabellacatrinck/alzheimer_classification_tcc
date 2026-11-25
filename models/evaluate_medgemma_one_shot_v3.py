"""
Avalia o MedGemma-4B-IT em modo ONE-SHOT (1 paciente-exemplo de cada classe)
para classificação binária CN vs AD — metodologicamente comparável ao
train_densenet_single.py: mesmo split train/test (sem cross-validation),
mesma fração de validação interna para calibrar threshold, avaliação final
única no test/.

Cada paciente-exemplo fornece suas 9 fatias; o prompt contém 9 imagens CN +
9 imagens AD + 1 imagem de consulta (19 imagens por chamada). Cada paciente
avaliado (val_internal e test/) é pontuado com TODAS as suas fatias, uma
chamada por fatia, e o score do paciente é a média. A calibração do
threshold roda sob exatamente as mesmas condições do teste (mesmo contexto de
exemplos, mesmas fatias por paciente, mesma agregação).

ESTE SCRIPT É AUTOCONTIDO: não importa nenhum outro script do projeto. A
lógica compartilhada com o zero-shot e o few-shot (famílias de tokens Yes/No,
agregação por paciente, busca de threshold, AUC com IC95%, prompts, split) foi
COPIADA literalmente para cá, marcada com "[CÓPIA do zero-shot]". Se alterar
algum desses trechos em outro script, altere aqui também, senão os
experimentos deixam de ser comparáveis.

O QUE MUDOU EM RELAÇÃO AO v3 ANTERIOR (que herdava o one-shot antigo):
  - O score é a diferença de log-probabilidades entre as FAMÍLIAS de tokens
    Yes/No (com e sem espaço). O modelo gera "No" sem espaço; os tokens
    " Yes"/" No" usados antes tinham massa ~1e-8. O script avisa se a massa
    na família Yes/No for baixa.
  - Score do paciente = média de d entre as fatias; threshold buscado no
    domínio do score (pontos médios entre scores únicos de val_internal),
    não mais numa grade 0.20-0.79; AUC por paciente com IC95% (bootstrap).
  - A pergunta é a do zero-shot (--prompt) + UMA referência aos "two examples
    above" (sem a palavra "new", que o zero-shot não tem). Asserts garantem
    que removendo a referência volta-se exatamente ao texto do zero-shot.
  - Barra de progresso por FATIA (com ETA) e um pré-voo que mede 2 chamadas
    reais, mostra tokens e pico de VRAM e projeta o tempo total antes de rodar
    val/test (--skip-preflight desliga).
  - --order ad-first inverte a ordem dos dois exemplos (controle de recência:
    o último exemplo antes da consulta é sempre da mesma classe).
  - Resultados em results_medgemma_oneshot_CN_vs_AD/v3/full_<prompt>[_adfirst]/.

FORMALIZAÇÃO DO PROMPT (com fontes, não inventado):

1. Estrutura do one-shot (como apresentar os exemplos e a imagem de
   consulta): Ferber et al., "In-context learning enables multimodal
   large language models to classify cancer pathology images", Nature
   Communications 2024 (arXiv:2403.07407). Testam explicitamente k=1
   (one-shot) com um VLM generativo (GPT-4V) em classificação binária de
   imagens médicas. Aqui, cada exemplo é um paciente por classe, apresentado
   por nove fatias com uma legenda, seguido da imagem de consulta.
   Achado central: zero-shot mal supera o acaso (56-62% de acurácia nos
   datasets binários do estudo), mas poucos exemplos já melhoram
   substancialmente o desempenho. Também usamos a técnica deles de
   enquadrar o contexto da tarefa explicitamente como pesquisa (não
   diagnóstico clínico real), que ajuda a mitigar recusas por alinhamento
   de segurança excessivo — adaptada aqui para ser uma descrição VERDADEIRA
   do seu estudo (dados ADNI de-identificados, pesquisa acadêmica), em vez
   do enquadramento "cenário hipotético, sem efeito no mundo real" que o
   estudo original usa para o GPT-4V.

2. Resposta forçada (Yes/No) + leitura dos logits do primeiro token:
   Farina et al. (2026, arXiv:2606.20852), que usa exatamente essa
   técnica com esse MESMO checkpoint (MedGemma-4B-IT, pesos congelados)
   para classificação binária em radiografias de tórax. (Aqui lida como
   diferença de log-probabilidades entre as famílias de tokens Yes/No.)

3. Âncora de critérios radiológicos (nomear achados específicos a
   procurar, em vez de perguntar "tem Alzheimer's?" sem contexto):
   Goyal et al. (2026, arXiv:2606.21861), que documentam formalmente que
   prompts "minimal" são mais propensos a colapso de classe em VLMs
   zero-shot. Os achados citados (atrofia hipocampal, aumento
   ventricular, afinamento cortical) são os MESMOS marcadores usados nos
   prompts do MedSigLIP na Tabela 2 da sua TCC I — mantém o vocabulário
   radiológico consistente entre os dois experimentos de VLM da tese.

Protocolo de split (idêntico ao train_densenet_single.py, mesmo SEED):
  train/ -> train_fit (exemplos one-shot + nada mais) / val_internal
  (calibra o threshold, nunca usado como exemplo) -> test/ (avaliado
  uma única vez, com o threshold já fixado). O script confere que os
  pacientes-exemplo não estão em val_internal nem em test/.

Requisitos:
    pip install transformers accelerate bitsandbytes torch pillow pandas numpy tqdm scikit-learn matplotlib seaborn

Uso:
    # Teste rápido primeiro (o pré-voo mostra tempo por chamada, tokens e VRAM):
    python evaluate_medgemma_one_shot_v3.py --prompt original --limit 3 --limit-val 6 --debug

    # Rodada completa (mesmo --prompt do zero-shot/few-shot):
    python evaluate_medgemma_one_shot_v3.py --prompt <escolhido>

    # Controle de recência e robustez à escolha dos pacientes-exemplo:
    python evaluate_medgemma_one_shot_v3.py --prompt <escolhido> --order ad-first
    python evaluate_medgemma_one_shot_v3.py --prompt <escolhido> --exemplar-seed 1
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

RESULTS_DIR = Path(f"results_medgemma_oneshot_{CLASS_1}_vs_{CLASS_2}") / "v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRACTION = 0.15  # idêntico ao train_densenet_single.py
MAX_SLICES_PER_PATIENT = 9  # o dataset tem 9 fatias por paciente
N_EXEMPLAR_SLICES = 9  # fatias de cada paciente-exemplo
SELECTION_METRIC = "balanced_accuracy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

N_BOOT = 2000  # reamostragens do bootstrap para o IC95% da AUC
N_INFERENCE_WARMUP = 2
N_INFERENCE_MEASURE = 5  # cada chamada tem 19 imagens — orçamento pequeno

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

# Legendas dos exemplos (o número segue a ordem de apresentação).
EXEMPLAR_CN_TEXT = (
    "Example {n}: these nine brain MRI slices are from a patient with preserved "
    "hippocampal volume and no neurodegenerative abnormalities "
    "(Cognitively Normal)."
)
EXEMPLAR_AD_TEXT = (
    "Example {n}: these nine brain MRI slices are from a patient with hippocampal "
    "atrophy and enlarged ventricles consistent with Alzheimer's "
    "disease."
)

REFERENCE_SENTENCE = "Use the two examples above as a reference."
ORIGINAL_REFERENCE_CLAUSE = "Comparing this slice to the two examples above, and considering"


def one_shot_query(prompt_name):
    """Pergunta do zero-shot + UMA referência aos dois exemplos."""
    base = PROMPTS[prompt_name]
    if prompt_name == "original":
        return base.replace("Considering these features,", f"{ORIGINAL_REFERENCE_CLAUSE} these features,")
    first, rest = base.split(". ", 1)
    return f"{first}. {REFERENCE_SENTENCE} {rest}"


def _check_prompt_consistency():
    """Removendo a referência aos exemplos, volta-se exatamente ao texto do zero-shot."""
    for name, base in PROMPTS.items():
        q = one_shot_query(name)
        if name == "original":
            stripped = q.replace(ORIGINAL_REFERENCE_CLAUSE, "Considering")
        else:
            stripped = q.replace(" " + REFERENCE_SENTENCE, "")
        assert stripped == base, f"O prompt '{name}' do one-shot difere do zero-shot além da referência aos exemplos."


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
# EXEMPLOS ONE-SHOT (1 paciente CN + 1 paciente AD, 9 fatias cada)
# ============================================================================

def select_exemplars(train_fit_records, seed=SEED):
    """MESMA regra do one-shot e do few-shot: um paciente CN e um AD sorteados de train_fit."""
    cn_candidates = [r for r in train_fit_records if r["label"] == CLASS_1]
    ad_candidates = [r for r in train_fit_records if r["label"] == CLASS_2]
    rng = random.Random(seed)
    return rng.choice(cn_candidates), rng.choice(ad_candidates)


def exemplar_slice_paths(record, n_slices=N_EXEMPLAR_SLICES):
    paths = sorted(Path(record["patient_dir"]).glob("*.npy"))
    if len(paths) < n_slices:
        raise ValueError(
            f"O paciente-exemplo {record['patient_id']} tem {len(paths)} fatias; "
            f"são necessárias pelo menos {n_slices}."
        )
    indices = np.linspace(0, len(paths) - 1, n_slices).round().astype(int)
    return [paths[i] for i in indices]


def build_exemplar_blocks(cn_images, ad_images, order="cn-first"):
    """Blocos (legenda + 9 imagens) dos dois exemplos, na ordem pedida."""
    if len(cn_images) != N_EXEMPLAR_SLICES or len(ad_images) != N_EXEMPLAR_SLICES:
        raise ValueError(f"Cada paciente-exemplo deve fornecer exatamente {N_EXEMPLAR_SLICES} imagens.")
    pair = [(EXEMPLAR_CN_TEXT, cn_images), (EXEMPLAR_AD_TEXT, ad_images)]
    if order == "ad-first":
        pair.reverse()
    blocks = []
    for n, (template, images) in enumerate(pair, start=1):
        blocks.append({"type": "text", "text": template.format(n=n)})
        blocks.extend({"type": "image", "image": image} for image in images)
    return blocks


def build_messages(ctx, query_image):
    content = ([{"type": "text", "text": RESEARCH_CONTEXT}] + ctx["blocks"]
               + [{"type": "text", "text": ctx["query"]}, {"type": "image", "image": query_image}])
    return [{"role": "user", "content": content}]


# ============================================================================
# SCORING ONE-SHOT
# ============================================================================

@torch.inference_mode()
def score_slice_one_shot(model, processor, query_image, ctx, tokens, verbose=False):
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


def count_slices(records, max_slices):
    return sum(len(select_slices(r["patient_dir"], max_slices=max_slices)) for r in records)


def evaluate_patient(model, processor, patient_dir, ctx, tokens, max_slices, verbose=False, pbar=None):
    rows = []
    for si, p in enumerate(select_slices(patient_dir, max_slices=max_slices)):
        r = score_slice_one_shot(model, processor, load_slice_as_image(p), ctx, tokens, verbose=verbose)
        r["slice_idx"] = si
        rows.append(r)
        if pbar is not None:
            pbar.update(1)
    return rows


def evaluate_population(model, processor, records, ctx, tokens, max_slices,
                        verbose_first_n=0, desc="Avaliando"):
    """Retorna (patient_df, slice_df). score do paciente = média de d entre as fatias.
    A barra avança por FATIA (cada fatia = 1 chamada ao modelo)."""
    label_map = {CLASS_1: 0, CLASS_2: 1}
    patient_rows, slice_rows = [], []
    pbar = tqdm(total=count_slices(records, max_slices), desc=desc, unit="fatia")
    try:
        for i, record in enumerate(records):
            verbose = i < verbose_first_n
            pbar.set_postfix(paciente=record["patient_id"])
            rows = evaluate_patient(model, processor, record["patient_dir"], ctx, tokens, max_slices,
                                    verbose=verbose, pbar=pbar)
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
                tqdm.write(f"  {record['patient_id']} (label={record['label']}) -> score={score:+.3f} "
                           f"(P={sigmoid(score):.4f})")
    finally:
        pbar.close()
    return pd.DataFrame(patient_rows), pd.DataFrame(slice_rows)


# ============================================================================
# PRÉ-VOO E TEMPO DE INFERÊNCIA
# ============================================================================

def preflight_timing(model, processor, ctx, tokens, first_record, max_slices, n_calls_total, n_calls=2):
    """Mede n_calls chamadas reais (a 1ª inclui aquecimento) e projeta o tempo total.
    Se o pico de VRAM alocada passar da VRAM física, o driver pode estar usando memória
    compartilhada (lentíssimo, sem erro)."""
    first_path = select_slices(first_record["patient_dir"], max_slices=max_slices)[0]
    image = load_slice_as_image(first_path)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    print("\n===== Pré-voo: medindo o tempo de chamadas reais =====")
    for i in range(n_calls):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        score_slice_one_shot(model, processor, image, ctx, tokens, verbose=True)  # imprime nº de tokens
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        print(f"  chamada {i + 1}/{n_calls}: {times[-1]:.1f} s")

    per_call = times[-1]  # a última, já sem o custo de aquecimento
    total_hours = per_call * n_calls_total / 3600
    print(f"  ≈ {per_call:.1f} s por chamada | {n_calls_total} chamadas (val + test, todas as fatias) "
          f"→ projeção ≈ {total_hours:.1f} h")

    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  pico de VRAM alocada: {peak:.2f} GB de {total:.2f} GB")
        if peak > 0.95 * total:
            print("  *** AVISO: o pico encostou/passou da VRAM física. No Windows o driver pode estar usando "
                  "memória compartilhada (RAM), o que deixa tudo muito lento sem dar erro. ***")
    if total_hours > 4:
        print("  *** AVISO: projeção acima de 4 h. Considere --skip-preflight só se aceitar esse tempo. ***")
    return per_call


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
                        help="mesma pergunta do zero-shot/few-shot (escolha UM, em val_internal)")
    parser.add_argument("--order", choices=["cn-first", "ad-first"], default="cn-first",
                        help="qual exemplo vem primeiro (controle de recência)")
    parser.add_argument("--exemplar-seed", type=int, default=SEED,
                        help="seed da escolha do par de pacientes-exemplo (padrão: SEED, o par original)")
    parser.add_argument("--limit", type=int, default=None, help="avaliar só os N primeiros pacientes do teste")
    parser.add_argument("--limit-val", type=int, default=None, help="usar só N pacientes de val_internal (teste rápido)")
    parser.add_argument("--debug", action="store_true",
                        help="imprime d/massa/token/nº de tokens para os 3 primeiros pacientes do teste")
    parser.add_argument("--max-slices", type=int, default=MAX_SLICES_PER_PATIENT)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-inference-timing", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="não medir o tempo de 2 chamadas antes de rodar val/test")
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
    if args.limit_val:
        val_internal = val_internal[: args.limit_val]
    print(f"\nSplit (idêntico ao train_densenet_single.py): {len(train_fit)} train_fit | "
          f"{len(val_internal)} val_internal (calibração) | {len(test_records)} test")

    cn_record, ad_record = select_exemplars(train_fit, seed=args.exemplar_seed)
    print(f"\nExemplo CN (one-shot, seed={args.exemplar_seed}): {cn_record['patient_id']}")
    print(f"Exemplo AD (one-shot, seed={args.exemplar_seed}): {ad_record['patient_id']}")

    # Proteção contra vazamento: exemplos só de train_fit, fora de val_internal e test/.
    other_ids = {r["patient_id"] for r in val_internal} | {r["patient_id"] for r in test_records}
    assert not ({cn_record["patient_id"], ad_record["patient_id"]} & other_ids), \
        "Paciente-exemplo também está em val_internal ou test/."

    cn_paths = exemplar_slice_paths(cn_record)
    ad_paths = exemplar_slice_paths(ad_record)
    print(f"Usando {N_EXEMPLAR_SLICES} fatias para cada paciente-exemplo "
          f"({2 * N_EXEMPLAR_SLICES} imagens-exemplo + 1 de consulta por chamada).")

    suffix = ("_adfirst" if args.order == "ad-first" else "") + \
             (f"_ex{args.exemplar_seed}" if args.exemplar_seed != SEED else "")
    out_dir = RESULTS_DIR / f"full_{args.prompt}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exemplars_used.json", "w") as f:
        json.dump({
            "prompt": args.prompt, "order": args.order, "exemplar_seed": args.exemplar_seed,
            "n_slices_per_exemplar": N_EXEMPLAR_SLICES,
            "cn_exemplar": cn_record["patient_id"], "cn_slices": [p.name for p in cn_paths],
            "ad_exemplar": ad_record["patient_id"], "ad_slices": [p.name for p in ad_paths],
        }, f, indent=2)

    model, processor = load_model_and_processor(quantize_4bit=not args.no_4bit)
    tokens = get_answer_token_families(processor)

    cn_images = [load_slice_as_image(p) for p in cn_paths]
    ad_images = [load_slice_as_image(p) for p in ad_paths]
    ctx = {"blocks": build_exemplar_blocks(cn_images, ad_images, order=args.order),
           "query": one_shot_query(args.prompt)}
    print(f"Pergunta: {ctx['query']}")

    mode_label = f"one-shot (9 fatias por exemplo, order={args.order}, prompt={args.prompt})"

    if not args.skip_preflight:
        n_calls_total = count_slices(val_internal, args.max_slices) + count_slices(test_records, args.max_slices)
        preflight_timing(model, processor, ctx, tokens, val_internal[0], args.max_slices, n_calls_total)

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
        json.dump({"prompt": args.prompt, "order": args.order, "exemplar_seed": args.exemplar_seed,
                   "n_slices_per_exemplar": N_EXEMPLAR_SLICES,
                   "threshold_score": threshold, "threshold_sigmoid": float(sigmoid(threshold)),
                   "n_patients_test": len(y_true), "val_median_mass_yes_no": float(val_mass),
                   "yes_token_ids": tokens["yes"], "no_token_ids": tokens["no"],
                   "metrics": metrics,
                   "cn_exemplar": cn_record["patient_id"], "ad_exemplar": ad_record["patient_id"]}, f, indent=4)

    report = classification_report(y_true, y_pred, target_names=[CLASS_1, CLASS_2], digits=4)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Mode: {mode_label}\nThreshold (score): {threshold:+.3f}\nN: {len(y_true)}\n\n{report}")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[CLASS_1, CLASS_2], yticklabels=[CLASS_1, CLASS_2])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — MedGemma one-shot 9 fatias [{args.prompt}] (n={len(y_true)})\n"
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