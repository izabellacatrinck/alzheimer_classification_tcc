#!/usr/bin/env python3
# ==============================================================================
# DATASET BUILDER - CORONAL SLICES (ADNI MRI)
# ==============================================================================

import shutil
import logging
import numpy as np
import cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================

INPUT_DIR = Path("Data/ADNI_PROCESSADO_4")

OUTPUT_DIR_NPY = Path("Data/ADNI_SAGITTAL_NPY")
OUTPUT_DIR_PNG = Path("Data/ADNI_SAGITTAL_PNG")

N_SLICES = 9

POSITION_TRIM_FRACTION = 0.15

MIN_BRAIN_FILL = 0.10
MIN_STD = 0.03
BRAIN_THRESHOLD = 0.08

SCORE_WEIGHTS = {
    "brain_fill": 0.35,
    "edge": 0.35,
    "variance": 0.20,
    "laplacian": 0.10,
}

CANDIDATE_POOL = N_SLICES * 4


# ==============================================================================
# LOAD + NORMALIZE
# ==============================================================================

def load_and_normalize(path: Path):
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr)

    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    return arr


# ==============================================================================
# INDEX (SAGITTAL)
# ==============================================================================
def extract_slice_index(path: Path):
    stem = path.stem

    if "slice_" in stem:
        try:
            return int(stem.split("slice_")[-1])
        except:
            pass

    tokens = stem.replace("-", "_").split("_")
    nums = [int(t) for t in tokens if t.isdigit()]
    return nums[-1] if nums else 0


# ==============================================================================
# HARD GATE
# ==============================================================================
def passes_hard_gate(arr):
    brain_mask = arr > BRAIN_THRESHOLD
    if np.mean(brain_mask) < MIN_BRAIN_FILL:
        return False
    if np.std(arr) < MIN_STD:
        return False
    return True


# ==============================================================================
# SCORE
# ==============================================================================
def compute_score(arr):

    fill = np.mean(arr > BRAIN_THRESHOLD)
    brain_fill_score = 1 - abs(fill - 0.55) / 0.55
    brain_fill_score = np.clip(brain_fill_score, 0, 1)

    img_u8 = (arr * 255).astype(np.uint8)

    edges = cv2.Canny(img_u8, 30, 80)
    edge_score = np.clip(np.mean(edges > 0) / 0.05, 0, 1)

    variance_score = np.clip(np.var(arr) / 0.06, 0, 1)

    lap = cv2.Laplacian(img_u8, cv2.CV_64F)
    lap_score = np.clip(np.var(lap) / 200, 0, 1)

    w = SCORE_WEIGHTS

    combined = (
        w["brain_fill"] * brain_fill_score +
        w["edge"] * edge_score +
        w["variance"] * variance_score +
        w["laplacian"] * lap_score
    )

    return combined


# ==============================================================================
# UNIFORM SAMPLING
# ==============================================================================
def uniform_sample(candidates, n):
    if len(candidates) <= n:
        return sorted(candidates, key=lambda x: x["idx"])

    candidates = sorted(candidates, key=lambda x: x["idx"])
    idxs = np.linspace(0, len(candidates)-1, n).astype(int)
    return [candidates[i] for i in idxs]


# ==============================================================================
# PROCESS PATIENT
# ==============================================================================
def process_patient(patient_dir):

    # 👇 SAGITTAL FOLDER (AJUSTE AQUI)
    sagittal_dir = patient_dir / "slices_entropy_sagittal"

    if not sagittal_dir.exists():
        return

    files = sorted(sagittal_dir.glob("*.npy"))

    if len(files) == 0:
        return

    slices = [{"path": f, "idx": extract_slice_index(f)} for f in files]
    slices.sort(key=lambda x: x["idx"])

    n = len(slices)
    trim = int(n * POSITION_TRIM_FRACTION)

    slices = slices[trim:n-trim]

    candidates = []

    for s in slices:
        arr = load_and_normalize(s["path"])

        if not passes_hard_gate(arr):
            continue

        s["score"] = compute_score(arr)
        candidates.append(s)

    if len(candidates) == 0:
        return

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    pool = candidates[:CANDIDATE_POOL]

    selected = uniform_sample(pool, N_SLICES)

    out_npy = OUTPUT_DIR_NPY / patient_dir.name
    out_png = OUTPUT_DIR_PNG / patient_dir.name

    out_npy.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    for s in selected:
        # Copia o .npy
        shutil.copy2(s["path"], out_npy / s["path"].name)

        # Se existir .png correspondente (gerado pelo step5), copie-o também
        src_png = s["path"].with_suffix(".png")
        if src_png.exists():
            shutil.copy2(src_png, out_png / src_png.name)
        else:
            logger.debug(f"{patient_dir.name}: PNG ausente: {src_png.name}")

    logger.info(f"{patient_dir.name}: {len(selected)} sagittal slices")


# ==============================================================================
# DATASET LOOP
# ==============================================================================
def process_dataset():

    OUTPUT_DIR_NPY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_PNG.mkdir(parents=True, exist_ok=True)

    patients = [p for p in INPUT_DIR.iterdir() if p.is_dir()]

    for p in patients:
        try:
            process_patient(p)
        except Exception as e:
            logger.error(f"{p.name}: {e}")


if __name__ == "__main__":
    process_dataset()