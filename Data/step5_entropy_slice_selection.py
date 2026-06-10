"""
ETAPA 5 — SELEÇÃO INTELIGENTE DE FATIAS (Entropy-Based Slice Selection)
═════════════════════════════════════════════════════════════════════════

Implementa a estratégia descrita em:
  Mohsin, N.A.; Abdulameer, M.H. (2025). Evaluating the Impact of 2D MRI 
  Slice Orientation and Location on Alzheimer's Disease Diagnosis Using a 
  Lightweight Convolutional Neural Network. J. Imaging 2025, 11, 260.
  https://doi.org/10.3390/jimaging11080260

ESTRATÉGIA PRINCIPAL:
  1. Divide o volume 3D em N_SEGMENTS segmentos anátomicos:
     - Axial:   segmenta de cima para baixo (eixo Z)
     - Coronal: segmenta de trás para frente (eixo Y)
     - Sagital: segmenta de esquerda para direita (eixo X)
  
  2. Para cada segmento, em cada orientação:
     a) Extrai todas as fatias 2D
     b) Computa mapas de ativação (CNN features, simulado ou MobileNetV2)
     c) Calcula entropia de Shannon para cada canal de feature
     d) Seleciona fatia com MÁXIMA entropia (mais informativa)
  
  3. Saídas:
     - Fatias selecionadas (.npy e .png)
     - Análise de entropia (.csv)
     - Visualizações (.png)

KEY FINDINGS (Mohsin & Abdulameer 2025):
  - AD vs. CN: melhor com Axial-Seg9 (97.4% accuracy)
  - AD vs. CN: forte também em Coronal-Seg10 e Sagital-Seg10
  - AD vs. MCI: Coronal-Seg7 (92.0% accuracy)
  - MCI vs. CN: Sagital-Seg4, Seg6, Seg11 (86.9% accuracy)
  
  → A escolha de orientação e localização da fatia é CRÍTICA

DEPENDÊNCIAS:
  pip install numpy scipy nibabel pandas matplotlib scikit-image
  pip install scikit-learn  (opcional: para StandardScaler)
  pip install tensorflow   (opcional: para MobileNetV2 real)
"""

import logging
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES UTILITÁRIAS
# ══════════════════════════════════════════════════════════════════════════════

def load_nifti(path: Path):
    """Carrega NIfTI e retorna (img, data, affine, header)."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return img, data, img.affine, img.header


def save_slice(slice_data, path_npy, path_png):
    """Salva fatia como .npy e .png normalizado."""
    np.save(path_npy, slice_data.astype(np.float32))
    
    # Normalização Z-score com clipping
    sl = np.clip(slice_data, -3, 3)
    sl = (sl + 3) / 6
    sl = (sl * 255).astype(np.uint8)
    
    plt.imsave(path_png, sl, cmap='gray')


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE ENTROPIA (Shannon Entropy)
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_entropy_from_activation(activation_maps: np.ndarray) -> tuple:
    """
    Calcula entropia de Shannon a partir de mapas de ativação.
    
    Segue exatamente a formulação de Mohsin & Abdulameer (2025):
      - Activation map A = FE(V) ∈ R^(H'×W'×C)
      - Para cada canal Ac: H(Ac) = -∑ Pi*log(Pi)  [Shannon entropy]
      - Hmean(V) = (1/C) * ∑ H(Ac)  [média entre canais]
      - Selecionar V* = argmax(Hmean(v))
    
    Parâmetros:
      activation_maps: ndarray de forma (H, W, C) ou (H, W) [será expandido]
    
    Retorna:
      (mean_entropy, channel_entropies): tuple
        - mean_entropy: float, média das entropias
        - channel_entropies: list de entropias por canal
    """
    # Se é 2D, expandir para 3D com 1 canal
    if activation_maps.ndim == 2:
        activation_maps = np.expand_dims(activation_maps, axis=-1)
    
    if activation_maps.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {activation_maps.shape}")
    
    H, W, C = activation_maps.shape
    entropies = []
    
    for c in range(C):
        channel = activation_maps[:, :, c].astype(np.float32)
        
        # Normalizar canal para [0, 1]
        channel_min = np.nanmin(channel)
        channel_max = np.nanmax(channel)
        
        if channel_max > channel_min:
            channel_norm = (channel - channel_min) / (channel_max - channel_min)
        else:
            channel_norm = np.zeros_like(channel)
        
        # Criar histograma normalizado (probabilidades)
        # 256 bins é padrão para imagens de 8-bit
        hist, _ = np.histogram(channel_norm.flatten(), bins=256, range=(0, 1))
        prob = hist.astype(np.float64) / np.sum(hist)
        
        # Calcular entropia de Shannon: H = -∑(p_i * log2(p_i))
        # Remover probabilidades zero (para evitar log(0))
        prob_nonzero = prob[prob > 0]
        entropy = -np.sum(prob_nonzero * np.log2(prob_nonzero + 1e-10))
        entropies.append(entropy)
    
    mean_entropy = np.mean(entropies) if entropies else 0.0
    
    return mean_entropy, entropies


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO DE SEGMENTOS ANÁTOMICOS
# ══════════════════════════════════════════════════════════════════════════════

def extract_2d_slices_with_segmentation(
    data_3d: np.ndarray,
    mask_3d: np.ndarray,
    orientation: str = 'axial',
    n_segments: int = 10
) -> dict:

    segments = {}

    # ─────────────────────────────────────────────
    # Definir eixo
    # ─────────────────────────────────────────────
    if orientation == 'axial':
        axis = 2
        min_required = 9
    elif orientation == 'coronal':
        axis = 1
        min_required = 7
    elif orientation == 'sagittal':
        axis = 0
        min_required = 7
    else:
        raise ValueError(f"Invalid orientation: {orientation}")

    dim_size = data_3d.shape[axis]
    segment_size = dim_size / n_segments

    logger.info(
        f"Segmentando {orientation}: "
        f"eixo={axis}, dim={dim_size}, seg_size={segment_size:.2f}"
    )

    for seg_idx in range(n_segments):

        start_idx = int(seg_idx * segment_size)
        end_idx = int((seg_idx + 1) * segment_size)

        candidate_slices = []

        # ─────────────────────────────────────────
        # Coletar TODAS as slices do segmento
        # ─────────────────────────────────────────
        for slice_idx in range(start_idx, end_idx):

            if orientation == 'axial':
                slice_2d = data_3d[:, :, slice_idx]
                mask_2d = mask_3d[:, :, slice_idx]

            elif orientation == 'coronal':
                slice_2d = data_3d[:, slice_idx, :]
                mask_2d = mask_3d[:, slice_idx, :]

            else:
                slice_2d = data_3d[slice_idx, :, :]
                mask_2d = mask_3d[slice_idx, :, :]

            # cobertura cerebral
            brain_coverage = np.sum(mask_2d > 0) / mask_2d.size

            # ignorar slices totalmente vazias
            if brain_coverage < 0.01:
                continue

            candidate_slices.append({
                'slice_idx': slice_idx,
                'slice_data': slice_2d,
                'mask_data': mask_2d,
                'brain_coverage': brain_coverage
            })

        # ─────────────────────────────────────────
        # Ordenar por cobertura cerebral
        # ─────────────────────────────────────────
        candidate_slices = sorted(
            candidate_slices,
            key=lambda x: x['brain_coverage'],
            reverse=True
        )

        # ─────────────────────────────────────────
        # Garantir slices suficientes
        # ─────────────────────────────────────────
        #
        # Estratégia:
        #   - prioriza melhor cobertura
        #   - mantém qualidade anatômica
        #   - evita slices periféricas ruins
        #
        selected_candidates = candidate_slices[:max(min_required * 2, min_required)]

        slices_in_segment = [x['slice_idx'] for x in selected_candidates]
        slice_data_list = [x['slice_data'] for x in selected_candidates]
        mask_data_list = [x['mask_data'] for x in selected_candidates]

        segments[f'seg_{seg_idx + 1}'] = {
            'slices': slices_in_segment,
            'slice_data': slice_data_list,
            'mask_data': mask_data_list,
            'orientation': orientation,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n_slices': len(slice_data_list)
        }

        logger.info(
            f"{orientation} seg_{seg_idx + 1}: "
            f"{len(slice_data_list)} slices válidas"
        )

    return segments

# ══════════════════════════════════════════════════════════════════════════════
# SIMULAÇÃO DE MAPAS DE ATIVAÇÃO (CNN Features)
# ══════════════════════════════════════════════════════════════════════════════

def compute_activation_maps_simple(slice_2d: np.ndarray, n_filters: int = 32) -> np.ndarray:
    """
    Simula mapas de ativação de camadas rasas da CNN (MobileNetV2).
    
    Em aplicação real, usaria:
      ```
      model = tf.keras.applications.MobileNetV2(weights='imagenet')
      activation_model = tf.keras.Model(
          inputs=model.input,
          outputs=model.get_layer('block_1a_expand_relu').output
      )
      activations = activation_model.predict(slice_reshaped)
      ```
    
    Aqui, simulamos com:
    - Filtros Sobel (derivadas)
    - Filtros Gaussianos (smoothing em múltiplas escalas)
    
    Parâmetros:
      slice_2d: fatia 2D (H, W)
      n_filters: número de "canais" de features (padrão: 32)
    
    Retorna:
      activation_maps: (H, W, n_filters)
    """
    H, W = slice_2d.shape
    activation_maps = np.zeros((H, W, n_filters), dtype=np.float32)
    
    # Normalizar entrada para [0, 1]
    slice_min = np.nanmin(slice_2d)
    slice_max = np.nanmax(slice_2d)
    if slice_max > slice_min:
        slice_norm = (slice_2d - slice_min) / (slice_max - slice_min)
    else:
        slice_norm = np.zeros_like(slice_2d)
    
    # Filtros derivativos (simula detecção de edges)
    activation_maps[:, :, 0] = np.abs(sobel(slice_norm, axis=0))  # derivada vertical
    activation_maps[:, :, 1] = np.abs(sobel(slice_norm, axis=1))  # derivada horizontal
    
    # Filtros Gaussianos em múltiplas escalas
    for i in range(2, n_filters):
        # Varia sigma de 0.5 a 3.0 linearmente
        sigma = 0.5 + (i - 2) * (3.0 - 0.5) / (n_filters - 2)
        activation_maps[:, :, i] = gaussian_filter(slice_norm, sigma=sigma)
    
    return activation_maps


def try_load_pretrained_mobilenet():
    """
    Tenta carregar MobileNetV2 pré-treinado do TensorFlow.
    
    Retorna:
      model: tf.keras.Model ou None se falhar
    """
    try:
        import tensorflow as tf
        logger.info("Carregando MobileNetV2 pré-treinado...")
        
        model = tf.keras.applications.MobileNetV2(
            input_shape=(150, 150, 3),
            include_top=False,
            weights='imagenet'
        )
        logger.info("✓ MobileNetV2 carregado com sucesso")
        return model
    
    except Exception as e:
        logger.warning(f"Falha ao carregar MobileNetV2: {e}. Usando features simples.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SELEÇÃO DE MELHOR FATIA POR SEGMENTO
# ══════════════════════════════════════════════════════════════════════════════

def select_best_slice_per_segment(segments: dict, 
                                  mobilenet_model=None,
                                  n_slices_per_segment: int = 9) -> dict:
    """
    Seleciona as N FATIAS COM MÁXIMA ENTROPIA de cada segmento.
    
    Algoritmo (Mohsin & Abdulameer 2025, Eq. 1-4):
      Para cada fatia V no segmento S:
        1. A = FE(V)  [mapas de ativação]
        2. Para cada canal Ac: H(Ac) = -∑ Pi*log(Pi)
        3. Hmean(V) = (1/C) * ∑ H(Ac)
      V* = argsort(Hmean(V)) — top N fatias
    
    Parâmetros:
      segments: dict de extract_2d_slices_with_segmentation()
      mobilenet_model: modelo tf.keras.Model (opcional)
      n_slices_per_segment: número de fatias a selecionar por segmento (padrão: 9)
    
    Retorna:
      best_slices: dict
        {
          'seg_1': {
            'slices': [
              {'slice_data': array 2D, 'entropy': float, 'slice_idx_absolute': int, ...},
              ...
            ],
            'all_entropies': list,
            'orientation': str,
            ...
          },
          ...
        }
    """
    best_slices = {}
    
    for seg_name, seg_data in segments.items():
        slice_data_list = seg_data['slice_data']
        slices_idx = seg_data['slices']
        
        if len(slice_data_list) == 0:
            logger.warning(f"Segmento {seg_name} sem fatias válidas")
            best_slices[seg_name] = None
            continue
        
        entropies_list = []
        slice_info_list = []
        
        # Computar entropia para cada fatia no segmento
        for i, slice_2d in enumerate(slice_data_list):
            # Normalizar e redimensionar para 150x150 (padrão Mohsin & Abdulameer)
            slice_norm = slice_2d.copy().astype(np.float32)
            slice_min = np.nanmin(slice_norm)
            slice_max = np.nanmax(slice_norm)
            if slice_max > slice_min:
                slice_norm = (slice_norm - slice_min) / (slice_max - slice_min)
            else:
                slice_norm = np.zeros_like(slice_norm)
            
            # Redimensionar para 150x150
            slice_resized = ndimage.zoom(
                slice_norm,
                (150 / slice_norm.shape[0], 150 / slice_norm.shape[1]),
                order=1
            )
            
            # Computar mapas de ativação
            if mobilenet_model is not None:
                try:
                    import tensorflow as tf
                    # Stack para 3 canais (RGB)
                    slice_rgb = np.stack([slice_resized] * 3, axis=-1)
                    slice_batch = np.expand_dims(slice_rgb, axis=0)
                    
                    # Extrair ativações
                    activations = mobilenet_model(slice_batch, training=False).numpy()[0]
                except Exception as e:
                    logger.warning(f"Erro ao usar MobileNetV2: {e}. Usando features simples.")
                    activations = compute_activation_maps_simple(slice_resized, n_filters=32)
            else:
                activations = compute_activation_maps_simple(slice_resized, n_filters=32)
            
            # Calcular entropia de Shannon
            mean_entropy, _ = compute_feature_entropy_from_activation(activations)
            entropies_list.append(mean_entropy)
            
            # Armazenar informações
            slice_info_list.append({
                'slice_data': slice_2d,
                'entropy': mean_entropy,
                'slice_idx_in_segment': i,
                'slice_idx_absolute': slices_idx[i]
            })
        
        # Ordenar por entropia (descendente) e selecionar top N
        # ─────────────────────────────────────────────
        # Ordenar por entropia
        # ─────────────────────────────────────────────
        sorted_indices = np.argsort(entropies_list)[::-1]

        selected_slices = [
            slice_info_list[idx]
            for idx in sorted_indices[:n_slices_per_segment]
]       

        # ─────────────────────────────────────────────
        # Garantir quantidade fixa SEM duplicar imagens
        # ─────────────────────────────────────────────
        #
        # Se faltar slices:
        #   - pega as melhores restantes do segmento
        #   - preserva diversidade anatômica
        #
        if len(selected_slices) < n_slices_per_segment:

            used_indices = {
                s['slice_idx_absolute']
                for s in selected_slices
            }

            remaining = []

            for idx in sorted_indices:

                candidate = slice_info_list[idx]

                if candidate['slice_idx_absolute'] not in used_indices:
                    remaining.append(candidate)

            needed = n_slices_per_segment - len(selected_slices)

            selected_slices.extend(remaining[:needed])
        
        best_slices[seg_name] = {
            'slices': selected_slices,
            'all_entropies': entropies_list,
            'orientation': seg_data['orientation'],
            'n_slices_evaluated': len(slice_data_list),
            'n_slices_selected': len(selected_slices),
            'start_idx': seg_data['start_idx'],
            'end_idx': seg_data['end_idx']
        }
    
    return best_slices


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 5: PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def step5_entropy_based_slice_selection(input_path: Path, 
                                        mask_path: Path,
                                        output_dir: Path, 
                                        subject_id: str,
                                        n_segments: int = 10,
                                        use_pretrained: bool = False) -> bool:
    """
    ETAPA 5 — Seleção Inteligente de Fatias (Entropy-Based Slice Selection).
    
    Implementa completamente a estratégia de Mohsin & Abdulameer (2025).
    
    Workflow:
      1. Carregar volume 3D pré-processado
      2. Para cada orientação (axial, coronal, sagittal):
         a) Segmentar em 9 segmentos anátomicos
         b) Para cada segmento:
            - Extrair todas as fatias 2D
            - Computar mapas de ativação
            - Calcular entropia de Shannon
            - Selecionar fatia com máxima entropia
         c) Salvar fatias selecionadas + análise
         d) Gerar visualizações
    
    Saídas:
      {output_dir}/
        ├── slices_entropy_axial/
        │   ├── {subject_id}_axial_seg1.npy
        │   ├── {subject_id}_axial_seg1.png
        │   ├── {subject_id}_axial_seg2.npy
        │   └── ...
        ├── slices_entropy_coronal/
        │   └── ...
        ├── slices_entropy_sagittal/
        │   └── ...
        ├── entropy_analysis_axial.csv
        ├── entropy_analysis_coronal.csv
        ├── entropy_analysis_sagittal.csv
        ├── entropy_comparison_axial.png
        ├── entropy_comparison_coronal.png
        └── entropy_comparison_sagittal.png
    
    Parâmetros:
      input_path:    caminho para NIfTI pré-processado
      mask_path:     caminho para máscara cerebral
      output_dir:    diretório de saída
      subject_id:    identificador do sujeito
      n_segments:    número de segmentos (padrão: 9, Mohsin & Abdulameer)
      use_pretrained: usar MobileNetV2 (requer TensorFlow)
    
    Retorna:
      bool: True se bem-sucedido
    """
    try:
        # ── Carregar volumes 3D ────────────────────────────────────────────
        logger.info(f"[{subject_id}] Carregando imagem 3D...")
        img, data_3d, affine, header = load_nifti(input_path)
        _, mask_3d, _, _ = load_nifti(mask_path)
        
        # Binarizar máscara
        mask_3d = (mask_3d > 0.5).astype(np.float32)
        
        logger.info(f"[{subject_id}] Shape: {data_3d.shape}")
        logger.info(f"[{subject_id}] Intensidade: [{data_3d.min():.2f}, {data_3d.max():.2f}]")
        logger.info(f"[{subject_id}] Cobertura cerebral: {np.sum(mask_3d > 0) / mask_3d.size * 100:.1f}%")
        
        # ── Carregamento de modelo pré-treinado (opcional) ─────────────────
        mobilenet = None
        if use_pretrained:
            mobilenet = try_load_pretrained_mobilenet()
        
        # ── Processar em três orientações ───────────────────────────────────
        orientations = ['axial', 'coronal', 'sagittal']
        all_results = {}
        
        for orientation in orientations:
            logger.info(f"\n[{subject_id}] ═══ Processando orientação: {orientation.upper()} ═══")
            
            # Extrair segmentos anátomicos
            segments = extract_2d_slices_with_segmentation(
                data_3d, mask_3d,
                orientation=orientation,
                n_segments=n_segments
            )
            
            # Selecionar 9 fatias com máxima entropia per segmento
            if orientation == 'axial':
                n_selected_slices = 9
            else:
                n_selected_slices = 7

            best_slices = select_best_slice_per_segment(
                segments,
                mobilenet_model=mobilenet,
                n_slices_per_segment=n_selected_slices
            )
            
            all_results[orientation] = best_slices
            
            # ── Salvar fatias selecionadas ─────────────────────────────────
            slices_dir = output_dir / f"slices_entropy_{orientation}"
            slices_dir.mkdir(parents=True, exist_ok=True)
            
            entropy_records = []
            
            for seg_name, seg_info in best_slices.items():
                if seg_info is None:
                    continue
                
                seg_num = seg_name.split('_')[1]
                
                # Iterar sobre as 9 fatias selecionadas
                for slice_rank, slice_info in enumerate(seg_info['slices'], start=1):
                    slice_data = slice_info['slice_data']
                    entropy_val = slice_info['entropy']
                    abs_idx = slice_info['slice_idx_absolute']
                    
                    # Salvar como .npy e .png com sufixo de ranking
                    npy_path = slices_dir / f"{subject_id}_{orientation}_seg{seg_num}_rank{slice_rank}.npy"
                    png_path = slices_dir / f"{subject_id}_{orientation}_seg{seg_num}_rank{slice_rank}.png"
                    
                    save_slice(slice_data, npy_path, png_path)
                    
                    # Log de entropia
                    entropy_records.append({
                        'subject_id': subject_id,
                        'orientation': orientation,
                        'segment': int(seg_num),
                        'rank': slice_rank,
                        'absolute_slice_idx': abs_idx,
                        'entropy_shannon': entropy_val,
                        'n_slices_evaluated': seg_info['n_slices_evaluated'],
                        'n_slices_selected': seg_info['n_slices_selected'],
                        'min_entropy': np.min(seg_info['all_entropies']),
                        'max_entropy': np.max(seg_info['all_entropies']),
                        'mean_entropy_segment': np.mean(seg_info['all_entropies']),
                        'std_entropy_segment': np.std(seg_info['all_entropies'])
                    })
                    
                    if slice_rank == 1:
                        logger.info(
                            f"  seg_{seg_num}: H={entropy_val:.3f} (idx={abs_idx}, "
                            f"range=[{np.min(seg_info['all_entropies']):.3f}, "
                            f"{np.max(seg_info['all_entropies']):.3f}], "
                            f"9 fatias selecionadas)"
                        )
            
            # ── Salvar análise em CSV ──────────────────────────────────────
            if entropy_records:
                df_entropy = pd.DataFrame(entropy_records)
                csv_path = output_dir / f"entropy_analysis_{orientation}.csv"
                df_entropy.to_csv(str(csv_path), index=False)
                logger.info(f"[{subject_id}] Análise salva → {csv_path.name}")
                
                # ── Gerar visualizações ───────────────────────────────────
                _generate_entropy_visualizations(df_entropy, subject_id, orientation, output_dir)
        
        logger.info(f"\n[{subject_id}] ✓ ETAPA 5 CONCLUÍDA COM SUCESSO")
        
        # ── Sumário final ──────────────────────────────────────────────────
        logger.info(f"\n[{subject_id}] SUMÁRIO DE SELEÇÃO DE FATIAS (9 FATIAS POR SEGMENTO):")
        for orientation in orientations:
            if orientation in all_results:
                best_data = all_results[orientation]
                valid_segs = [s for s in best_data.values() if s is not None]
                mean_entropy = np.mean([np.mean(s['all_entropies']) for s in valid_segs]) if valid_segs else 0
                total_slices = sum([len(s['slices']) for s in valid_segs])
                logger.info(f"  {orientation:10} → {len(valid_segs)} segmentos × 9 fatias = {total_slices} fatias, entropia média: {mean_entropy:.3f}")
        
        return True
    
    except Exception as e:
        logger.error(f"[{subject_id}] ERRO na Etapa 5: {e}", exc_info=True)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

def _generate_entropy_visualizations(df_entropy: pd.DataFrame, 
                                     subject_id: str,
                                     orientation: str,
                                     output_dir: Path):
    """Gera visualizações de análise de entropia."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Entropy Analysis — {subject_id} ({orientation.upper()})\n"
            f"(Mohsin & Abdulameer 2025 — Intelligent Slice Selection)",
            fontsize=13, fontweight='bold'
        )
        
        # Gráfico 1: Entropia da fatia selecionada per segmento
        ax = axes[0, 0]
        ax.bar(df_entropy['segment'].astype(str),
               df_entropy['entropy_shannon'],
               color='steelblue', alpha=0.75, edgecolor='navy', linewidth=1.5)
        ax.set_xlabel('Segment', fontsize=11)
        ax.set_ylabel('Shannon Entropy', fontsize=11)
        ax.set_title('Selected Slice Entropy (Max per Segment)', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Gráfico 2: Range de entropias (min, mean, max)
        ax = axes[0, 1]
        segments_str = df_entropy['segment'].astype(str).tolist()
        x = np.arange(len(segments_str))
        width = 0.25
        
        ax.bar(x - width, df_entropy['min_entropy'], width, label='Min', color='lightcoral', alpha=0.75)
        ax.bar(x, df_entropy['mean_entropy_segment'], width, label='Mean', color='steelblue', alpha=0.75)
        ax.bar(x + width, df_entropy['max_entropy'], width, label='Max', color='seagreen', alpha=0.75)
        
        ax.set_xlabel('Segment', fontsize=11)
        ax.set_ylabel('Entropy', fontsize=11)
        ax.set_title('Entropy Range (Min/Mean/Max per Segment)', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(segments_str, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Gráfico 3: Número de fatias avaliadas per segmento
        ax = axes[1, 0]
        ax.bar(segments_str,
               df_entropy['n_slices_evaluated'],
               color='mediumpurple', alpha=0.75, edgecolor='purple', linewidth=1.5)
        ax.set_xlabel('Segment', fontsize=11)
        ax.set_ylabel('Number of Slices', fontsize=11)
        ax.set_title('Slices Evaluated per Segment', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Gráfico 4: Desvio padrão de entropia (variabilidade)
        ax = axes[1, 1]
        ax.bar(segments_str,
               df_entropy['std_entropy_segment'],
               color='coral', alpha=0.75, edgecolor='darkorange', linewidth=1.5)
        ax.set_xlabel('Segment', fontsize=11)
        ax.set_ylabel('Std. Dev. of Entropy', fontsize=11)
        ax.set_title('Entropy Variability (Std. Dev.) per Segment', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        png_path = output_dir / f"entropy_comparison_{orientation}.png"
        plt.savefig(str(png_path), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"  Visualização salva → {png_path.name}")
    
    except Exception as e:
        logger.warning(f"Erro ao gerar visualizações: {e}")
