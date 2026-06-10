#!/usr/bin/env python3
# ==============================================================================
# SELEÇÃO DE SLICES MRI PARA CLASSIFICAÇÃO DE ALZHEIMER
# ==============================================================================
#
# PROBLEMA CORRIGIDO:
#   O código anterior selecionava slices periféricos/vazios (como tecido
#   de couro cabeludo) por não ter critérios robustos de qualidade anatômica.
#
# ESTRATÉGIA ADOTADA (baseada em literatura recente 2023-2025):
#
#   1. CORTE POSICIONAL HARD (mais importante)
#      → Descartar frações extremas do volume (topo/base)
#      → Garante que apenas a região cerebral central seja candidata
#      Ref: Şener et al. (2025), Scientific Reports
#           "first and last 20 slices discarded since they contained
#            little brain tissue"
#
#   2. SCORING MULTICRITÉRIO por slice
#      → brain_fill_score:  fração de pixels acima de threshold
#      → edge_score:        riqueza de bordas anatômicas (Canny)
#      → variance_score:    variância normalizada (complexidade estrutural)
#      → laplacian_score:   nitidez/foco do slice
#      → combined_score:    média ponderada
#      Ref: Şener et al. (2025) — Canny edge count para rich anatomical content
#           Tran et al. (2018) — Shannon image entropy para slice informativeness
#
#   3. SELEÇÃO UNIFORME NO ESPAÇO ANATÔMICO
#      → Sobre os N candidatos top, amostrar uniformemente pelo índice axial
#      → Garante cobertura de diferentes regiões cerebrais
#      → Evita N slices muito próximos (redundantes)
#      Ref: Choudhury et al. (2024), Information Fusion
#           "16 most significant 2D slices from axial views after
#            rigorous comparative analysis"
#
# RESULTADO ESPERADO:
#   Slices com tecido cerebral real, boa cobertura anatômica,
#   estruturas visíveis (ventrículos, hipocampo, sulcos corticais),
#   adequados para treinar DenseNet/EfficientNet.
#
# ==============================================================================

import shutil
import logging
import numpy as np
import cv2

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================

INPUT_DIR  = Path("Data/ADNI_PROCESSADO_4")
OUTPUT_DIR_PNG = Path("Data/ADNI_AXIAL_PNG")
OUTPUT_DIR_NPY = Path("Data/ADNI_AXIAL_NPY_15")

# Quantos slices selecionar por paciente
N_SLICES = 10

# ── Corte posicional ──────────────────────────────────────────────────────────
# Descarta as frações extremas do eixo axial antes de qualquer scoring.
# 0.15 = ignora os 15% inferiores (queixo/pescoço) e 15% superiores (topo do crânio).
# Ajuste: aumentar se ainda aparecerem periféricos; diminuir se cortar estruturas.
POSITION_TRIM_FRACTION = 0.15

# ── Filtros mínimos (hard gate — elimina antes do scoring) ────────────────────
# Fração mínima de pixels considerados "cérebro" (> BRAIN_THRESHOLD)
MIN_BRAIN_FILL    = 0.10   # 10 % da imagem deve ser tecido cerebral
# Desvio padrão mínimo — evita slices homogêneos/brancos/pretos
MIN_STD           = 0.03
# Limiar para classificar pixel como "tecido" (imagem normalizada 0-1)
BRAIN_THRESHOLD   = 0.08

# ── Pesos do score combinado ──────────────────────────────────────────────────
# brain_fill:  cobertura de tecido cerebral (quanto do slice é cérebro)
# edge:        densidade de bordas Canny (riqueza de estruturas anatômicas)
# variance:    variância espacial (complexidade de textura)
# laplacian:   nitidez (foco)
SCORE_WEIGHTS = {
    "brain_fill": 0.35,
    "edge":       0.35,
    "variance":   0.20,
    "laplacian":  0.10,
}

# ── Pool de candidatos ────────────────────────────────────────────────────────
# Quantos dos melhores slices (por score) passam para a etapa de
# amostragem uniforme. Deve ser >= N_SLICES * 2 para dar margem.
CANDIDATE_POOL = N_SLICES * 4

# ==============================================================================
# CARREGAMENTO E NORMALIZAÇÃO
# ==============================================================================

def load_and_normalize(path: Path) -> np.ndarray:
    """
    Carrega .npy, substitui NaN/Inf e normaliza para [0, 1].
    """
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    return arr


# ==============================================================================
# EXTRAÇÃO DO ÍNDICE AXIAL
# ==============================================================================

def extract_slice_index(path: Path) -> int:
    """
    Extrai o índice axial do nome do arquivo.
    Suporta padrões: slice_42.npy, slice42.npy, patient_000_042.npy, etc.
    """
    stem = path.stem

    # padrão mais comum: slice_<n>
    if "slice_" in stem:
        try:
            return int(stem.split("slice_")[-1])
        except ValueError:
            pass

    # fallback: último número no nome
    tokens = stem.replace("-", "_").split("_")
    numbers = [int(t) for t in tokens if t.isdigit()]
    return numbers[-1] if numbers else 0


# ==============================================================================
# HARD GATE — FILTRO MÍNIMO
# ==============================================================================

def passes_hard_gate(arr: np.ndarray) -> bool:
    """
    Rejeita slices periféricos/vazios antes de qualquer scoring.

    Critérios (todos devem ser satisfeitos):
      1. Fração de pixels considerados "tecido" >= MIN_BRAIN_FILL
      2. Desvio padrão global >= MIN_STD
      3. Slice não é completamente homogêneo
    """
    brain_mask = arr > BRAIN_THRESHOLD
    fill = float(np.mean(brain_mask))

    if fill < MIN_BRAIN_FILL:
        return False

    if float(np.std(arr)) < MIN_STD:
        return False

    return True


# ==============================================================================
# SCORING MULTICRITÉRIO
# ==============================================================================

def compute_slice_score(arr: np.ndarray) -> dict:
    """
    Calcula 4 scores normalizados [0, 1] para um slice.

    Returns:
        dict com scores individuais e 'combined' (média ponderada).
    """

    # ── 1. Brain Fill Score ────────────────────────────────────────────────────
    # Fração de pixels que são tecido cerebral.
    # Slices centrais têm fill alto; periféricos têm fill baixo.
    # Penaliza tanto fill muito baixo (periferia) quanto fill = 1.0
    # (slice saturado / artefato).
    fill = float(np.mean(arr > BRAIN_THRESHOLD))

    # Curva triangular: máximo em fill = 0.55, cai em direção a 0 e 1.
    # Ajuste o pico conforme seu dataset (0.55 é bom para axial ADNI).
    brain_fill_score = 1.0 - abs(fill - 0.55) / 0.55
    brain_fill_score = float(np.clip(brain_fill_score, 0.0, 1.0))

    # ── 2. Edge Score (Canny) ─────────────────────────────────────────────────
    # Conta bordas anatômicas usando Canny.
    # Ref: Şener et al. (2025) — "slice with the highest number of edges
    # was selected as the subject-level reference image"
    img_u8 = (arr * 255).astype(np.uint8)
    edges = cv2.Canny(img_u8, threshold1=30, threshold2=80)
    edge_density = float(np.mean(edges > 0))

    # Normaliza pelo máximo teórico razoável (5 % de pixels em borda é muito rico)
    edge_score = float(np.clip(edge_density / 0.05, 0.0, 1.0))

    # ── 3. Variance Score ─────────────────────────────────────────────────────
    # Variância espacial indica complexidade estrutural.
    # Slices com hipocampo, ventrículos, sulcos têm alta variância.
    variance = float(np.var(arr))

    # Normaliza: variância acima de 0.06 já é score 1.0
    variance_score = float(np.clip(variance / 0.06, 0.0, 1.0))

    # ── 4. Laplacian Score (Nitidez) ──────────────────────────────────────────
    # Mede o foco do slice.
    # Slices desfocados (bordas do volume) têm laplaciano baixo.
    lap = cv2.Laplacian(img_u8, cv2.CV_64F)
    sharpness = float(np.var(lap))

    # Normaliza: acima de 200 já é score 1.0
    laplacian_score = float(np.clip(sharpness / 200.0, 0.0, 1.0))

    # ── Score Combinado ────────────────────────────────────────────────────────
    w = SCORE_WEIGHTS
    combined = (
        w["brain_fill"] * brain_fill_score
        + w["edge"]      * edge_score
        + w["variance"]  * variance_score
        + w["laplacian"] * laplacian_score
    )

    return {
        "brain_fill": brain_fill_score,
        "edge":       edge_score,
        "variance":   variance_score,
        "laplacian":  laplacian_score,
        "combined":   float(combined),
    }


# ==============================================================================
# AMOSTRAGEM UNIFORME NO ESPAÇO ANATÔMICO
# ==============================================================================

def uniform_anatomical_sample(
    candidates: list,
    n_slices: int,
) -> list:
    """
    Dado um pool de candidatos (já filtrados e ordenados por índice axial),
    retorna n_slices slices espaçados uniformemente no eixo anatômico.

    Isso garante cobertura de diferentes regiões cerebrais e evita
    redundância por slices muito próximos.

    Args:
        candidates: lista de dicts com chaves 'path', 'idx', 'scores'
        n_slices:   quantos slices retornar

    Returns:
        lista com n_slices elementos, ordenados por índice axial
    """
    if len(candidates) == 0:
        return []

    if len(candidates) <= n_slices:
        return sorted(candidates, key=lambda x: x["idx"])

    # Ordenar por índice axial para amostragem espacialmente uniforme
    candidates_sorted = sorted(candidates, key=lambda x: x["idx"])

    # linspace garante espaçamento uniforme
    pick_indices = np.linspace(0, len(candidates_sorted) - 1, n_slices)
    pick_indices = np.round(pick_indices).astype(int)

    selected = [candidates_sorted[i] for i in pick_indices]

    return selected


# ==============================================================================
# PROCESSO POR PACIENTE
# ==============================================================================

def process_patient(patient_dir: Path):
    """
    Pipeline completo para um paciente:

    1. Coleta todos os .npy da pasta axial
    2. Aplica corte posicional (descarta extremidades)
    3. Aplica hard gate (descarta slices vazios/periféricos)
    4. Calcula scores multicritério nos candidatos restantes
    5. Seleciona o pool top-K por score combinado
    6. Amostra N_SLICES uniformemente no espaço anatômico
    7. Copia arquivos para OUTPUT_DIR
    """

    patient_id = patient_dir.name
    axial_dir  = patient_dir / "slices_entropy_axial"

    if not axial_dir.exists():
        logger.warning(f"[{patient_id}] pasta axial não encontrada — pulando")
        return

    npy_files = sorted(axial_dir.glob("*.npy"))
    npy_files = [p for p in npy_files if "comparison" not in p.name.lower()]

    if len(npy_files) == 0:
        logger.warning(f"[{patient_id}] nenhum .npy encontrado — pulando")
        return

    # ── Passo 1: extrair índices e ordenar ─────────────────────────────────
    all_slices = []
    for p in npy_files:
        all_slices.append({"path": p, "idx": extract_slice_index(p)})

    all_slices.sort(key=lambda x: x["idx"])
    n_total = len(all_slices)

    # ── Passo 2: corte posicional ───────────────────────────────────────────
    # Descarta POSITION_TRIM_FRACTION do início e do fim da pilha axial.
    trim = int(np.ceil(n_total * POSITION_TRIM_FRACTION))
    trim = max(trim, 1)  # pelo menos 1 slice cortado em cada ponta

    trimmed = all_slices[trim : n_total - trim]

    if len(trimmed) == 0:
        logger.warning(
            f"[{patient_id}] corte posicional removeu todos os slices "
            f"(total={n_total}, trim={trim}) — pulando"
        )
        return

    # ── Passo 3: hard gate + scoring ───────────────────────────────────────
    candidates = []

    for item in trimmed:
        try:
            arr = load_and_normalize(item["path"])
        except Exception as e:
            logger.debug(f"[{patient_id}] erro ao carregar {item['path'].name}: {e}")
            continue

        # Hard gate: descarta slices vazios/periféricos rapidamente
        if not passes_hard_gate(arr):
            continue

        scores = compute_slice_score(arr)
        item["scores"] = scores
        candidates.append(item)

    if len(candidates) == 0:
        logger.warning(
            f"[{patient_id}] nenhum candidato passou o hard gate — pulando"
        )
        return

    # ── Passo 4: selecionar pool top-K por score combinado ─────────────────
    # Ordena por score combinado decrescente e pega os melhores CANDIDATE_POOL.
    # Esse pool ainda mantém diversidade de posição axial para a etapa seguinte.
    candidates_sorted_by_score = sorted(
        candidates,
        key=lambda x: x["scores"]["combined"],
        reverse=True,
    )

    pool = candidates_sorted_by_score[:CANDIDATE_POOL]

    # ── Passo 5: amostragem uniforme no eixo anatômico ─────────────────────
    selected = uniform_anatomical_sample(pool, N_SLICES)

    if len(selected) < N_SLICES:
        logger.warning(
            f"[{patient_id}] apenas {len(selected)}/{N_SLICES} slices disponíveis"
        )
        # Não pula: usa o que tem (pode acontecer em volumes pequenos)

    # ── Passo 6: copiar para output ────────────────────────────────────────
    out_npy = OUTPUT_DIR_NPY / patient_id
    out_png = OUTPUT_DIR_PNG / patient_id
    out_npy.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    for item in selected:
        src_npy = item["path"]
        src_png = src_npy.with_suffix(".png")

        shutil.copy2(src_npy, out_npy / src_npy.name)

        if src_png.exists():
            shutil.copy2(src_png, out_png / src_png.name)
        else:
            logger.debug(f"[{patient_id}] PNG ausente: {src_png.name}")

    # Log detalhado dos slices selecionados
    indices_str = ", ".join(str(s["idx"]) for s in selected)
    scores_str  = ", ".join(f"{s['scores']['combined']:.3f}" for s in selected)
    logger.info(
        f"[{patient_id}] {len(selected)} slices selecionados | "
        f"índices: [{indices_str}] | "
        f"scores: [{scores_str}]"
    )


# ==============================================================================
# PROCESSO DO DATASET COMPLETO
# ==============================================================================

def process_dataset():

    OUTPUT_DIR_PNG.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_NPY.mkdir(parents=True, exist_ok=True)

    patients = sorted(p for p in INPUT_DIR.iterdir() if p.is_dir())
    logger.info(f"{len(patients)} pacientes encontrados em {INPUT_DIR}")

    ok = 0
    skipped = 0

    for patient_dir in patients:
        try:
            before = sum(1 for _ in (OUTPUT_DIR_NPY / patient_dir.name).glob("*.npy")) \
                     if (OUTPUT_DIR_NPY / patient_dir.name).exists() else 0

            process_patient(patient_dir)

            after = sum(1 for _ in (OUTPUT_DIR_NPY / patient_dir.name).glob("*.npy")) \
                    if (OUTPUT_DIR_NPY / patient_dir.name).exists() else 0

            if after > before:
                ok += 1
            else:
                skipped += 1

        except Exception as e:
            logger.error(f"[{patient_dir.name}] erro inesperado: {e}", exc_info=True)
            skipped += 1

    logger.info(
        f"Dataset finalizado — "
        f"{ok} pacientes processados, {skipped} pulados."
    )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    process_dataset()
    logger.info("DONE")