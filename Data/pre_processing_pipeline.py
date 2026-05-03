"""
Pipeline de Pré-processamento MRI - 4 Etapas
============================================
Etapas:
  1. Motion Correction   — FSL FLIRT (registro rígido 6 DOF ao MNI152)
  2. Skull Stripping      — HD-BET (deep learning, robusto para AD/MCI/CN)
  3. Intensity Normalization — ANTs N4 bias field + Z-score (numpy)
  4. Background Removal   — threshold + crop via nibabel/numpy

Quality Control automático após cada etapa:
  - SNR intra-máscara
  - CNR (GM vs WM)
  - Brain mask coverage (% voxels cerebrais)
  - Coeficiente de variação de intensidade (CV)
  - Relatório CSV consolidado por sujeito

Referências:
  - Henschel et al. 2020 (FastSurfer)
  - Isensee et al. 2019 (HD-BET)
  - Tustison et al. 2010 (N4ITK)
  - Wen et al. 2020 (data leakage MRI)
  - Hendriks et al. 2024 (QC T1w review)

Uso:
  python mri_preprocessing_pipeline.py \
      --input_dir /data/ADNI/raw \
      --output_dir /data/ADNI/preprocessed \
      --subject_list subjects.txt \
      --mni_template /usr/share/fsl/data/standard/MNI152_T1_1mm.nii.gz \
      --n_jobs 4

Dependências:
  pip install nibabel numpy scipy pandas matplotlib antspyx
  pip install hd-bet  (ou: pip install hd_bet)
  FSL instalado e no PATH (fsl.fmrib.ox.ac.uk)
"""

from importlib.resources import files
import os
import sys
import argparse
import logging
import subprocess
import json
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from nibabel.processing import resample_from_to

# ─── Configuração de logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: str, subject_id: str = "") -> bool:
    """Executa comando shell com log. Retorna True se bem-sucedido."""
    logger.info(f"[{subject_id}] CMD: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"[{subject_id}] ERRO: {result.stderr.strip()}")
        return False
    return True


def load_nifti(path: Path):
    """Carrega NIfTI e retorna (img, data, affine, header)."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return img, data, img.affine, img.header


def save_nifti(data: np.ndarray, affine: np.ndarray, header, out_path: Path):
    """Salva array como NIfTI."""
    out_img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(out_img, str(out_path))

def resample_mask_to_image(mask_path: Path, ref_path: Path, output_path: Path):
    import nibabel as nib
    from nibabel.processing import resample_from_to

    mask_img = nib.load(str(mask_path))
    ref_img = nib.load(str(ref_path))

    resampled = resample_from_to(mask_img, ref_img, order=0)  # nearest neighbor!
    nib.save(resampled, str(output_path))


def save_slice(slice_data, path_npy, path_png):
    np.save(path_npy, slice_data.astype(np.float32))

    # 🔥 normalização consistente com Z-score
    sl = np.clip(slice_data, -3, 3)
    sl = (sl + 3) / 6
    sl = (sl * 255).astype(np.uint8)

    plt.imsave(path_png, sl, cmap='gray')

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — MOTION CORRECTION (FSL FLIRT, registro rígido 6 DOF → MNI152)
# ══════════════════════════════════════════════════════════════════════════════

def step1_motion_correction(input_path: Path, output_path: Path,
                             mni_template: Path, subject_id: str) -> bool:
    """
    Registro rígido (6 DOF) ao MNI152 1mm para corrigir orientação/movimento.

    Literatura: Smith et al. 2004 (FSL); Jenkinson & Smith 2001 (FLIRT)
    Para T1w estrutural do ADNI, 6 DOF é suficiente — corrige translações
    e rotações sem deformar a geometria cerebral.
    """
    mat_path = output_path.parent / f"{subject_id}_mc.mat"

    cmd = (
        f"flirt "
        f"-in {input_path} "
        f"-ref {mni_template} "
        f"-out {output_path} "
        f"-omat {mat_path} "
        f"-dof 6 "
        f"-interp trilinear "
        f"-cost mutualinfo "
        f"-searchrx -30 30 -searchry -30 30 -searchrz -30 30"
    )
    success = run_cmd(cmd, subject_id)
    if success:
        logger.info(f"[{subject_id}] Etapa 1 concluída → {output_path.name}")
    return success


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — SKULL STRIPPING (HD-BET)
# ══════════════════════════════════════════════════════════════════════════════

def step2_skull_stripping(input_path: Path, output_path: Path,
                           subject_id: str) -> bool:
    """
    Skull stripping com HD-BET (deep learning, Isensee et al. 2019).

    HD-BET supera FSL BET em imagens com atrofia severa (AD/MCI),
    sendo especialmente robusto ao ADNI multi-scanner.

    Saídas geradas pelo HD-BET:
      - {output_path}       → imagem skull-stripped
      - {output_path}_mask  → brain mask binária

    Literatura: Isensee et al. 2019, Human Brain Mapping
    """
    # HD-BET gera automaticamente a máscara junto com a imagem
    cmd = (
    f"hd-bet "
    f"-i {input_path} "
    f"-o {output_path} "
    f"-device cuda "
    f"--disable_tta "
    f"--save_bet_mask"
)
    success = run_cmd(cmd, subject_id)
    if success:
        logger.info(f"[{subject_id}] Etapa 2 concluída → {output_path.name}")
    return success


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — INTENSITY NORMALIZATION (ANTs N4 + Z-score)
# ══════════════════════════════════════════════════════════════════════════════

def step3_intensity_normalization(input_path: Path, mask_path: Path,
                                   output_path: Path, subject_id: str) -> bool:
    """
    Normalização em dois passos:
      3a. Correção de bias field com N4BiasFieldCorrection (ANTs)
          → Remove não-uniformidades de intensidade introduzidas pelo scanner
          → Literatura: Tustison et al. 2010 (N4ITK), IEEE Trans Med Imaging
      3b. Z-score intra-máscara (media=0, std=1)
          → Padroniza escala de intensidade entre sujeitos e scanners
          → Literatura: Zhang et al. 2024 (ADNI Alzheimer classification)

    A combinação N4 + Z-score é o padrão ouro para pipelines multi-scanner.
    """
    # ── 3a. N4 Bias Field Correction via ANTsPy ────────────────────────────
    try:
        import ants
        img_ants = ants.image_read(str(input_path))
        mask_ants = ants.image_read(str(mask_path))

        n4_result = ants.n4_bias_field_correction(
            img_ants,
            mask=mask_ants,
            shrink_factor=4,          # 4 = bom balanço velocidade/precisão
            convergence={'iters': [50, 50, 50, 50], 'tol': 1e-07},
            spline_param=200
        )
        n4_path = output_path.parent / f"{subject_id}_n4.nii.gz"
        ants.image_write(n4_result, str(n4_path))
        logger.info(f"[{subject_id}] N4 bias correction concluída")

    except ImportError:
        # Fallback: N4 via linha de comando ANTs
        n4_path = output_path.parent / f"{subject_id}_n4.nii.gz"
        cmd = (
            f"N4BiasFieldCorrection "
            f"-d 3 "
            f"-i {input_path} "
            f"-x {mask_path} "
            f"-o {n4_path} "
            f"--shrink-factor 4 "
            f"--convergence [50x50x50x50,1e-7]"
        )
        if not run_cmd(cmd, subject_id):
            return False

    # ── 3b. Z-score intra-máscara ─────────────────────────────────────────
    img_n4, data_n4, affine, header = load_nifti(n4_path)
    _, mask_data, _, _ = load_nifti(mask_path)

    # Garante máscara binária (HD-BET pode gerar probabilística)
    mask_bin = (mask_data > 0.5).astype(np.uint8)
    brain_voxels = data_n4[mask_bin == 1]

    # 🔥 ROBUST NORMALIZATION
    p2, p98 = np.percentile(brain_voxels, [2, 98])
    data_n4 = np.clip(data_n4, p2, p98)

    brain_voxels = data_n4[mask_bin == 1]
    mu = brain_voxels.mean()
    sigma = brain_voxels.std()

    if sigma < 1e-6:
        logger.warning(f"[{subject_id}] Desvio padrão muito baixo ({sigma:.4f}), verificar imagem!")
        return False

    data_norm = np.copy(data_n4)
    data_norm[mask_bin == 1] = (data_n4[mask_bin == 1] - mu) / sigma
    data_norm[mask_bin == 0] = 0
    data_norm = np.clip(data_norm, -3, 3)

    save_nifti(data_norm, affine, header, output_path)
    logger.info(f"[{subject_id}] Etapa 3 concluída → Z-score (μ={mu:.1f}, σ={sigma:.1f})")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — BACKGROUND REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def step4_background_removal(input_path: Path, mask_path: Path,
                               output_path: Path, subject_id: str) -> bool:
    """
    Remoção de fundo em dois passos:
      4a. Aplicação da brain mask → zera voxels fora do cérebro
      4b. Crop do FOV → remove fatias/linhas completamente vazias

    Garante que o volume final contenha apenas tecido cerebral,
    reduzindo custo computacional no treinamento da CNN.

    Literatura: Zhang et al. 2024; Soomro et al. 2025 (MDPI Diagnostics)
    """
    _, data, affine, header = load_nifti(input_path)
    _, mask, _, _ = load_nifti(mask_path)

    # ── 4a. Aplica máscara ─────────────────────────────────────────────────
    mask_bin = (mask > 0.5).astype(np.uint8)
    data_masked = data * mask_bin
    brain_coords = np.where(mask_bin == 1)
    if len(brain_coords[0]) == 0:
        logger.error(f"[{subject_id}] Máscara vazia! Verificar skull stripping.")
        return False

    x_min, x_max = brain_coords[0].min(), brain_coords[0].max() + 1
    y_min, y_max = brain_coords[1].min(), brain_coords[1].max() + 1
    z_min, z_max = brain_coords[2].min(), brain_coords[2].max() + 1

    # Margem de segurança de 5 voxels
    pad = 5
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    z_min = max(0, z_min - pad)
    x_max = min(data.shape[0], x_max + pad)
    y_max = min(data.shape[1], y_max + pad)
    z_max = min(data.shape[2], z_max + pad)

    data_cropped = data_masked[x_min:x_max, y_min:y_max, z_min:z_max]

    # Atualiza affine para refletir o crop
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, :3] @ np.array([x_min, y_min, z_min]) + affine[:3, 3]

    new_img = nib.Nifti1Image(data_cropped, new_affine)
    nib.save(new_img, str(output_path))
    logger.info(
        f"[{subject_id}] Etapa 4 concluída → shape {data.shape} → {data_cropped.shape}"
    )
    return True

def step4b_resample_to_standard(input_path: Path,
                               output_path: Path,
                               mni_template: Path,
                               target_shape=(160, 192, 160),
                               subject_id: str = "") -> bool:
    try:
        img = nib.load(str(input_path))

        # 🔥 Carrega o MNI template
        mni_img = nib.load(str(mni_template))

        # 🔥 Usa o affine REAL do MNI
        target = (mni_img.shape, mni_img.affine)

        resampled = resample_from_to(img, target, order=1)

        nib.save(resampled, str(output_path))

        logger.info(f"[{subject_id}] Resample correto (MNI-aligned) concluído")
        return True

    except Exception as e:
        logger.error(f"[{subject_id}] Erro no resample: {e}")
        return False

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 5 — SLICE EXTRACTION (20 slices por plano)
# ══════════════════════════════════════════════════════════════════════════════

def step5_slice_extraction(input_path: Path,
                          mask_path: Path,
                          output_dir: Path,
                          subject_id: str,
                          n_slices: int = 20):
    """
    NOVO: Slice extraction baseada em conteúdo cerebral

    - Seleciona apenas slices com cérebro suficiente
    - Evita slices vazios ou irrelevantes
    """

    _, data, _, _ = load_nifti(input_path)
    _, mask, _, _ = load_nifti(mask_path)

    def select_informative_slices(mask_volume, axis, n_slices):
        scores = []

        for i in range(mask_volume.shape[axis]):
            if axis == 0:
                slice_mask = mask_volume[i, :, :]
            elif axis == 1:
                slice_mask = mask_volume[:, i, :]
            else:
                slice_mask = mask_volume[:, :, i]

            brain_pixels = np.sum(slice_mask > 0.5)
            scores.append((i, brain_pixels))

        scores.sort(key=lambda x: x[1], reverse=True)

        # 🔥 DISTRIBUIÇÃO ao invés de pegar só os top
        selected_positions = np.linspace(0, len(scores)-1, n_slices).astype(int)
        selected = [scores[i][0] for i in selected_positions]

        return sorted(selected)

    axial_idx = select_informative_slices(mask, axis=2, n_slices=n_slices)
    coronal_idx = select_informative_slices(mask, axis=1, n_slices=n_slices)
    sagittal_idx = select_informative_slices(mask, axis=0, n_slices=n_slices)

    axial_dir = output_dir / "axial"
    coronal_dir = output_dir / "coronal"
    sagittal_dir = output_dir / "sagittal"

    axial_dir.mkdir(exist_ok=True)
    coronal_dir.mkdir(exist_ok=True)
    sagittal_dir.mkdir(exist_ok=True)

    for i, idx in enumerate(axial_idx):
        sl = data[:, :, idx]

        save_slice(
            sl,
            axial_dir / f"{subject_id}_axial_{i:02d}.npy",
            axial_dir / f"{subject_id}_axial_{i:02d}.png"
        )

    for i, idx in enumerate(coronal_idx):
        sl = data[:, idx, :]
        save_slice(
            sl,
            coronal_dir / f"{subject_id}_coronal_{i:02d}.npy",
            coronal_dir / f"{subject_id}_coronal_{i:02d}.png"
        )

    for i, idx in enumerate(sagittal_idx):
        sl = data[idx, :, :]
        save_slice(
            sl,
            sagittal_dir / f"{subject_id}_sagittal_{i:02d}.npy",
            sagittal_dir / f"{subject_id}_sagittal_{i:02d}.png"
        )

    logger.info(f"[{subject_id}] Slice extraction inteligente concluída")


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CONTROL — MÉTRICAS E RELATÓRIO
# ══════════════════════════════

def compute_qc_metrics(processed_path: Path, mask_path: Path,
                        subject_id: str, label: str) -> dict:
    """
    Calcula métricas de QC conforme Hendriks et al. 2024 (Neuroradiology)
    e o sistema LONI QC (SNR, CNR, CV, coverage).

    Métricas calculadas:
      - SNR: Signal-to-Noise Ratio intra-máscara
      - CNR: Contrast-to-Noise Ratio (aproximado GM vs WM)
      - CV:  Coeficiente de variação intra-máscara
      - brain_coverage: fração de voxels com sinal dentro da máscara
      - mean_intensity: intensidade média intra-máscara
      - std_intensity: desvio padrão intra-máscara
      - nonzero_fraction: fração de voxels não-zero no volume completo
    """
    metrics = {
        'subject_id': subject_id,
        'stage': label,
        'file': processed_path.name,
        'status': 'failed'
    }

    if not processed_path.exists() or not mask_path.exists():
        return metrics

    try:
        _, data, _, _ = load_nifti(processed_path)
        _, mask, _, _ = load_nifti(mask_path)

        mask_bin = (mask > 0.5).astype(np.uint8)
        brain = data[mask_bin == 1]
        background = data[mask_bin == 0]

        if len(brain) == 0:
            metrics['status'] = 'empty_mask'
            return metrics

        mean_brain = brain.mean()
        std_brain = brain.std()
        std_bg = background.std() if len(background) > 0 else 1.0

        # SNR = média do cérebro / std do background
        snr = mean_brain / (std_brain + 1e-8)

        # CNR aproximado: contraste GM/WM via limiar de Otsu simplificado
        p33 = np.percentile(brain, 33)
        p66 = np.percentile(brain, 66)
        gm_approx = brain[(brain >= p33) & (brain < p66)]
        wm_approx = brain[brain >= p66]
        cnr = (wm_approx.mean() - gm_approx.mean()) / (std_bg + 1e-8) \
              if len(gm_approx) > 0 and len(wm_approx) > 0 else 0.0

        # CV = coeficiente de variação
        cv = (std_brain / (abs(mean_brain) + 1e-8)) * 100

        # Brain coverage = voxels com sinal > 0 dentro da máscara
        coverage = np.sum(data[mask_bin == 1] > 0) / (np.sum(mask_bin) + 1e-8)

        # Fração não-zero do volume completo
        nonzero_fraction = (data > 0).sum() / data.size

        metrics.update({
            'status': 'ok',
            'snr': round(float(snr), 3),
            'cnr': round(float(cnr), 3),
            'cv_pct': round(float(cv), 3),
            'brain_coverage': round(float(coverage), 4),
            'mean_intensity': round(float(mean_brain), 4),
            'std_intensity': round(float(std_brain), 4),
            'nonzero_fraction': round(float(nonzero_fraction), 4),
            'voxel_count_brain': int(len(brain)),
            'shape': str(data.shape),
        })

    except Exception as e:
        metrics['status'] = f'error: {str(e)}'

    return metrics


def flag_qc_failures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica critérios de exclusão conforme literatura (Hendriks et al. 2024).
    Sinaliza sujeitos que precisam de inspeção manual.
    """
    df = df.copy()

    # Thresholds baseados na literatura e práticas ADNI/UK Biobank
    df['flag_low_snr']      = df['snr'] < 5.0           # SNR muito baixo
    df['flag_low_coverage'] = df['brain_coverage'] < 0.85  # cobertura < 85%
    df['flag_high_cv']      = df['cv_pct'] > 50.0       # CV > 50% = heterogêneo
    df['flag_low_cnr']      = df['cnr'] < 1.0            # CNR insuficiente
    df['flag_failed']       = df['status'] != 'ok'

    flag_cols = ['flag_low_snr', 'flag_low_coverage', 'flag_high_cv',
                 'flag_low_cnr', 'flag_failed']
    df['needs_review'] = df[flag_cols].any(axis=1)

    return df


def save_qc_mosaic(processed_path: Path, subject_id: str, stage: str,
                    out_dir: Path):
    """
    Salva mosaico de slices (axial, coronal, sagital) para inspeção visual.
    Conforme recomendação de inspeção visual pós-processamento (Di & Biswal 2023).
    """
    if not processed_path.exists():
        return

    try:
        _, data, _, _ = load_nifti(processed_path)
        mid = [s // 2 for s in data.shape]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='black')
        slices = [
            data[mid[0], :, :],
            data[:, mid[1], :],
            data[:, :, mid[2]],
        ]
        titles = ['Axial', 'Coronal', 'Sagital']

        for ax, sl, title in zip(axes, slices, titles):
            vmin, vmax = np.percentile(sl[sl != 0], [2, 98]) if sl.any() else (0, 1)
            ax.imshow(np.rot90(sl), cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}", color='white', fontsize=10)
            ax.axis('off')

        fig.suptitle(f"{subject_id} — {stage}", color='white', fontsize=11)
        plt.tight_layout()
        out_path = out_dir / f"{subject_id}_{stage}_mosaic.png"
        plt.savefig(str(out_path), dpi=100, bbox_inches='tight', facecolor='black')
        plt.close()

    except Exception as e:
        logger.warning(f"[{subject_id}] Erro ao gerar mosaico: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE POR SUJEITO
# ══════════════════════════════════════════════════════════════════════════════

def process_subject(args_tuple) -> dict:
    """
    Executa o pipeline completo para um único sujeito.
    Retorna dicionário com métricas de QC de todas as etapas.
    """
    subject_id, input_file, output_dir, mni_template, generate_mosaics = args_tuple

    subj_dir = Path(output_dir) / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = subj_dir / 'qc_mosaics'
    if generate_mosaics:
        qc_dir.mkdir(exist_ok=True)

    input_path = Path(input_file)
    all_metrics = []

    # ── Caminhos de saída por etapa ────────────────────────────────────────
    p = {
        'mc':     subj_dir / f"{subject_id}_mc.nii.gz",
        'ss':     subj_dir / f"{subject_id}_ss.nii.gz",
        'ss_mask':subj_dir / f"{subject_id}_ss_mask.nii.gz",
        'norm':   subj_dir / f"{subject_id}_norm.nii.gz",
        'final':  subj_dir / f"{subject_id}_final.nii.gz",
    }

    # ── Etapa 1: Motion Correction ─────────────────────────────────────────
    logger.info(f"[{subject_id}] === Iniciando Etapa 1: Motion Correction ===")
    ok1 = step1_motion_correction(input_path, p['mc'], Path(mni_template), subject_id)
    if not ok1:
        return {'subject_id': subject_id, 'status': 'FAILED_step1', 'metrics': []}

    if generate_mosaics:
        save_qc_mosaic(p['mc'], subject_id, 'step1_mc', qc_dir)

    # ── Etapa 2: Skull Stripping ───────────────────────────────────────────
    logger.info(f"[{subject_id}] === Iniciando Etapa 2: Skull Stripping ===")
    ok2 = step2_skull_stripping(p['mc'], p['ss'], subject_id)
    if not ok2:
        return {'subject_id': subject_id, 'status': 'FAILED_step2', 'metrics': []}

    # HD-BET gera a máscara com sufixo _mask.nii.gz automaticamente
    # Detecta automaticamente a máscara gerada pelo HD-BET
    # Detecta automaticamente a máscara gerada pelo HD-BET (compatível com várias versões)
    mask_candidates = sorted([
        f for f in subj_dir.glob("*.nii*")
        if (
            subject_id in f.name
            and "ss" in f.name
            and ("mask" in f.name.lower() or "bet" in f.name.lower())
        )
    ])

    if len(mask_candidates) == 0:
        logger.error(f"[{subject_id}] Máscara HD-BET não encontrada!")
        logger.error(f"[{subject_id}] Arquivos disponíveis: {[f.name for f in subj_dir.glob('*')]}")
        return {'subject_id': subject_id, 'status': 'FAILED_step2_mask', 'metrics': []}

    # Pega a máscara encontrada
    found_mask = mask_candidates[0]

    # Padroniza nome da máscara
    standard_mask = subj_dir / f"{subject_id}_ss_mask.nii.gz"

    # Renomeia (ou substitui se já existir)
    if found_mask != standard_mask:
        found_mask.rename(standard_mask)

    p['ss_mask'] = standard_mask

    logger.info(f"[{subject_id}] Máscara detectada: {found_mask.name} → {standard_mask.name}")

    # 🔍 Validação da máscara (CRÍTICO)
    _, mask_data, _, _ = load_nifti(p['ss_mask'])

# 🔹 Cria máscara binária (AGORA mask_bin existe)
    mask_bin = (mask_data > 0.5).astype(np.uint8)

    # 🔹 Validação: máscara vazia
    if np.sum(mask_bin) == 0:
        logger.error(f"[{subject_id}] Máscara vazia!")
        return {'subject_id': subject_id, 'status': 'FAILED_empty_mask', 'metrics': all_metrics}

    # 🔹 Validação: proporção de cérebro
    brain_ratio = np.sum(mask_bin) / mask_bin.size

    if brain_ratio < 0.15 or brain_ratio > 0.5:
        logger.error(f"[{subject_id}] Máscara inválida! brain_ratio={brain_ratio:.3f}")
        return {'subject_id': subject_id, 'status': 'FAILED_bad_mask', 'metrics': all_metrics}

    qc2 = compute_qc_metrics(p['ss'], p['ss_mask'], subject_id, 'step2_skull_strip')
    all_metrics.append(qc2)
    if generate_mosaics:
        save_qc_mosaic(p['ss'], subject_id, 'step2_ss', qc_dir)

    # ── Etapa 3: Intensity Normalization ──────────────────────────────────
    logger.info(f"[{subject_id}] === Iniciando Etapa 3: Intensity Normalization ===")
    ok3 = step3_intensity_normalization(p['ss'], p['ss_mask'], p['norm'], subject_id)
    if not ok3:
        return {'subject_id': subject_id, 'status': 'FAILED_step3', 'metrics': all_metrics}

    qc3 = compute_qc_metrics(p['norm'], p['ss_mask'], subject_id, 'step3_norm')
    all_metrics.append(qc3)
    if generate_mosaics:
        save_qc_mosaic(p['norm'], subject_id, 'step3_norm', qc_dir)

    # ── Etapa 4: Background Removal ───────────────────────────────────────
    logger.info(f"[{subject_id}] === Iniciando Etapa 4: Background Removal ===")
    ok4 = step4_background_removal(p['norm'], p['ss_mask'], p['final'], subject_id)
    if not ok4:
        return {'subject_id': subject_id, 'status': 'FAILED_step4', 'metrics': all_metrics}

    # 🔥 NOVA ETAPA 4B: Padronização 3D
    p['resampled'] = subj_dir / f"{subject_id}_resampled.nii.gz"

    ok4b = step4b_resample_to_standard(
    input_path=p['final'],
    output_path=p['resampled'],
    mni_template=Path(mni_template),
    subject_id=subject_id
)
    if not ok4b:
        return {'subject_id': subject_id, 'status': 'FAILED_step4b', 'metrics': all_metrics}
    
    p['resampled_mask'] = subj_dir / f"{subject_id}_resampled_mask.nii.gz"

    resample_mask_to_image(
        p['ss_mask'],
        p['resampled'],
        p['resampled_mask']
    )

    qc4 = compute_qc_metrics(
    p['resampled'],
    p['resampled_mask'],
    subject_id,
    'step4_resampled'
)
    all_metrics.append(qc4)
    if generate_mosaics:
        save_qc_mosaic(p['resampled'], subject_id, 'step4_resampled', qc_dir)

    # ── Etapa 5: Slice Extraction ─────────────────────────────────────────────
    logger.info(f"[{subject_id}] === Iniciando Etapa 5: Slice Extraction ===")
    step5_slice_extraction(
    input_path=p['resampled'],
    mask_path=p['resampled_mask'],
    output_dir=subj_dir,
    subject_id=subject_id
)

    logger.info(f"[{subject_id}] Pipeline concluído com sucesso → {p['final'].name}")
    return {'subject_id': subject_id, 'status': 'SUCCESS', 'metrics': all_metrics}


# ══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO FINAL DE QC
# ══════════════════════════════════════════════════════════════════════════════

def generate_qc_report(all_results: list, output_dir: Path):
    """
    Gera relatório consolidado de QC em CSV e PNG.
    """
    report_dir = output_dir / 'QC_report'
    report_dir.mkdir(exist_ok=True)

    # Coleta todas as métricas
    rows = []
    for result in all_results:
        for m in result.get('metrics', []):
            rows.append(m)

    if not rows:
        logger.warning("Nenhuma métrica de QC coletada.")
        return

    df = pd.DataFrame(rows)
    df = flag_qc_failures(df)

    # ── CSV consolidado ────────────────────────────────────────────────────
    csv_path = report_dir / 'qc_metrics.csv'
    df.to_csv(str(csv_path), index=False)
    logger.info(f"Relatório CSV salvo → {csv_path}")

    # ── Lista de sujeitos para revisão manual ─────────────────────────────
    needs_review = df[df['needs_review'] == True]['subject_id'].unique()
    review_path = report_dir / 'subjects_needs_review.txt'
    with open(str(review_path), 'w') as f:
        for s in needs_review:
            f.write(s + '\n')
    logger.info(f"Sujeitos para revisão ({len(needs_review)}): {review_path}")

    # ── Gráficos de distribuição das métricas ─────────────────────────────
    final_df = df[df['stage'] == 'step4_resampled'].copy()
    if len(final_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QC Report — Distribuição das Métricas (Etapa Final)', fontsize=13)

    metrics_to_plot = [
        ('snr',           'SNR intra-máscara',            'steelblue',  5.0),
        ('cnr',           'CNR (GM vs WM aprox.)',         'seagreen',   1.0),
        ('cv_pct',        'Coeficiente de Variação (%)',   'coral',      50.0),
        ('brain_coverage','Brain mask coverage',           'mediumpurple',0.85),
    ]

    for ax, (col, title, color, threshold) in zip(axes.flat, metrics_to_plot):
        if col not in final_df.columns:
            continue
        vals = final_df[col].dropna()
        ax.hist(vals, bins=30, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
                   label=f'Threshold={threshold}')
        n_flagged = (vals < threshold).sum() if col != 'cv_pct' else (vals > threshold).sum()
        ax.set_title(f"{title}\n(n={len(vals)}, flagged={n_flagged})", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel('Sujeitos')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = report_dir / 'qc_distributions.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráficos de QC salvos → {fig_path}")

    # ── Sumário no terminal ────────────────────────────────────────────────
    total = final_df['subject_id'].nunique()
    ok = (final_df['needs_review'] == False).sum()
    review = (final_df['needs_review'] == True).sum()

    print("\n" + "="*60)
    print("  SUMÁRIO DO PIPELINE DE PRÉ-PROCESSAMENTO")
    print("="*60)
    print(f"  Total de sujeitos processados : {total}")
    print(f"  Aprovados automaticamente     : {ok}")
    print(f"  Para revisão manual           : {review}")
    print(f"\n  Métricas médias (etapa final):")
    for col in ['snr', 'cnr', 'cv_pct', 'brain_coverage']:
        if col in final_df.columns:
            print(f"    {col:<20} = {final_df[col].mean():.3f} ± {final_df[col].std():.3f}")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("SCRIPT INICIOU")
    parser = argparse.ArgumentParser(
        description='Pipeline de pré-processamento MRI (4 etapas) com QC automático'
    )
    parser.add_argument('--input_dir',     required=True,
                        help='Diretório com NIfTI originais (um por sujeito)')
    parser.add_argument('--output_dir',    required=True,
                        help='Diretório de saída do pipeline')
    parser.add_argument('--subject_list',  required=False, default=None,
                        help='Arquivo .txt com IDs de sujeitos (um por linha). '
                             'Se omitido, processa todos os .nii.gz encontrados.')
    parser.add_argument('--mni_template',  required=True,
                        help='Caminho para MNI152_T1_1mm.nii.gz')
    parser.add_argument('--n_jobs',        type=int, default=1,
                        help='Número de sujeitos em paralelo (default: 1)')
    parser.add_argument('--no_mosaics',    action='store_true',
                        help='Desativa geração de mosaicos de QC visual')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Coleta lista de sujeitos ───────────────────────────────────────────
    if args.subject_list:
        with open(args.subject_list) as f:
            subjects = [line.strip() for line in f if line.strip()]
        subject_files = []
        for sid in subjects:
            candidates = list(Path(args.input_dir).rglob(f"{sid}/*.nii*"))
            if candidates:
                subject_files.append((sid, str(candidates[0])))
            else:
                logger.warning(f"Arquivo não encontrado para sujeito: {sid}")
    else:
        nii_files = sorted(Path(args.input_dir).rglob('*.nii*'))
        subject_files = [(f.parent.name, str(f)) for f in nii_files]

    logger.info(f"Total de sujeitos a processar: {len(subject_files)}")

    # ── Processa em paralelo ───────────────────────────────────────────────
    tasks = [
        (sid, fpath, str(output_dir), args.mni_template, not args.no_mosaics)
        for sid, fpath in subject_files
    ]

    all_results = []
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {executor.submit(process_subject, t): t[0] for t in tasks}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    status = result.get('status', 'UNKNOWN')
                    logger.info(f"[{sid}] Status final: {status}")
                except Exception as e:
                    logger.error(f"[{sid}] Exceção inesperada: {e}")
                    all_results.append({'subject_id': sid, 'status': f'EXCEPTION: {e}', 'metrics': []})
    else:
        for task in tasks:
            result = process_subject(task)
            all_results.append(result)

    # ── Gera relatório de QC ───────────────────────────────────────────────
    generate_qc_report(all_results, output_dir)

    # ── Salva log de status geral ──────────────────────────────────────────
    status_log = output_dir / 'pipeline_status.json'
    summary = [{'subject_id': r['subject_id'], 'status': r['status']} for r in all_results]
    with open(str(status_log), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Status geral salvo → {status_log}")


if __name__ == '__main__':
    main()
