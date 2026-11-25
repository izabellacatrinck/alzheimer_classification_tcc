"""
Diagnóstico do zero-shot MedGemma (CN vs AD).

REGRA: tudo aqui roda em val_internal (mesmo split, mesmo SEED) — NUNCA no
test/, para não "espiar" o conjunto de teste enquanto você decide mudanças
de prompt/extração.

Modo 1 (sem GPU) — olhar os resultados que você já tem:
    python diagnose_zero_shot.py --analyze-csv \
        results_medgemma_zeroshot_CN_vs_AD/predictions_test.csv \
        results_medgemma_oneshot_CN_vs_AD/predictions_test.csv \
        results_medgemma_fewshot_CN_vs_AD/predictions_test.csv

Modo 2 (GPU) — diagnóstico completo do zero-shot:
    python diagnose_zero_shot.py --n-per-class 10
    python diagnose_zero_shot.py --n-per-class 5 --variants original minimal   # rápido

Fica na mesma pasta de evaluate_medgemma_zero_shot.py (importa dele config,
prompts, carregamento de slices e do modelo — assim o diagnóstico mede
exatamente o mesmo pipeline).

Saídas em results_medgemma_zeroshot_CN_vs_AD/diagnostics/:
    slice_scores.csv, describe.txt, montage.png
"""

import argparse
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import evaluate_zero_shot as zs

YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO", " NO"]

DESCRIBE_PROMPT = (
    "Describe this image. What imaging modality and anatomical plane is it, "
    "and which brain structures are visible? Is the hippocampus visible?"
)
MINIMAL_PROMPT = (
    "Does this brain MRI slice show signs of Alzheimer's disease? "
    "Answer with only Yes or No."
)


# ============================================================================
# Estatística: AUC com IC95% por bootstrap (reamostragem estratificada)
# ============================================================================

def auc_safe(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, np.asarray(s))


def bootstrap_auc(y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


# ============================================================================
# MODO 1 — analisar CSVs existentes (sem GPU)
# ============================================================================

def analyze_csv(paths):
    for p in paths:
        df = pd.read_csv(p)
        y, s = df["y_true"].values, df["probability"].values
        auc = roc_auc_score(y, s)
        lo, hi = bootstrap_auc(y, s)
        print(f"\n{p}")
        print(f"  n={len(df)} (AD={int(y.sum())}, CN={int((1 - y).sum())})  "
              f"AUC={auc:.3f}  IC95%=[{lo:.3f}, {hi:.3f}]")
        print(f"  probability: min={s.min():.2e}  mediana={np.median(s):.2e}  max={s.max():.4f}")
        print(f"  <0.20: {(s < 0.20).mean():.0%} | entre 0.20 e 0.80: "
              f"{((s >= 0.20) & (s <= 0.80)).mean():.0%} | >0.80: {(s > 0.80).mean():.0%}")


# ============================================================================
# MODO 2 — diagnóstico com o modelo
# ============================================================================

def build_messages(variant, image):
    img = {"type": "image", "image": image}

    def txt(t):
        return {"type": "text", "text": t}

    if variant == "original":          # idêntico ao script zero-shot
        content = [txt(zs.RESEARCH_CONTEXT), txt(zs.QUERY_PROMPT), img]
    elif variant == "image_first":     # imagem antes do texto
        content = [img, txt(zs.RESEARCH_CONTEXT), txt(zs.QUERY_PROMPT)]
    elif variant == "no_context":      # sem o enquadramento de pesquisa
        content = [txt(zs.QUERY_PROMPT), img]
    elif variant == "minimal":         # pergunta curta, sem âncora radiológica
        content = [txt(MINIMAL_PROMPT), img]
    else:
        raise ValueError(variant)
    return [{"role": "user", "content": content}]


def token_ids(tokenizer, variants):
    out = {}
    for v in variants:
        t = tokenizer.encode(v, add_special_tokens=False)
        if len(t) == 1:
            out[v] = t[0]
    return out


@torch.inference_mode()
def next_token_logprobs(model, processor, messages):
    # mesmo mecanismo do script original: generate(max_new_tokens=1) + scores[0]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=zs.DTYPE)
    out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    return torch.log_softmax(out.scores[0][0].float(), dim=-1)


def score_row(logp, tokenizer, yes_ids, no_ids, id_ad_orig, id_cn_orig):
    ly = torch.logsumexp(logp[yes_ids], dim=0).item()
    ln = torch.logsumexp(logp[no_ids], dim=0).item()
    top1 = int(torch.argmax(logp).item())
    return {
        # o que o script original mede (tokens ' Yes' / ' No' com espaço)
        "d_orig": (logp[id_ad_orig] - logp[id_cn_orig]).item(),
        "mass_orig": float(np.exp(logp[id_ad_orig].item()) + np.exp(logp[id_cn_orig].item())),
        # família de tokens Yes/No (com e sem espaço, maiúscula/minúscula)
        "d_family": ly - ln,
        "mass_family": float(np.exp(np.logaddexp(ly, ln))),
        "top1": tokenizer.decode([top1]),
        "top1_id": top1,
    }


def sample_val_patients(n_per_class, seed):
    all_train = zs.load_records(zs.TRAIN_LABELS, zs.TRAIN_DIR, "train")
    labels = [r["label"] for r in all_train]
    _, val_internal = train_test_split(
        all_train, test_size=zs.VAL_FRACTION, stratify=labels, random_state=zs.SEED
    )
    rng = random.Random(seed)
    chosen = []
    for cls in (zs.CLASS_1, zs.CLASS_2):
        pool = [r for r in val_internal if r["label"] == cls]
        chosen += rng.sample(pool, min(n_per_class, len(pool)))
    print(f"\nDiagnóstico em {len(chosen)} pacientes de val_internal "
          f"({n_per_class} por classe, no máximo)")
    return chosen


def summarize(df):
    print("\n" + "=" * 78)
    print("RESUMO POR VARIANTE DE PROMPT (val_internal, subamostra)")
    print("=" * 78)
    for v, g in df.groupby("variant"):
        print(f"\n--- variante: {v} ---")
        for col in ("d_orig", "d_family"):
            slice_auc = auc_safe(g["y_true"], g[col])
            pt = g.groupby("patient_id").agg(y=("y_true", "first"), s=(col, "mean"))
            p_auc = auc_safe(pt["y"], pt["s"])
            lo, hi = bootstrap_auc(pt["y"].values, pt["s"].values)
            print(f"  {col:<9} AUC fatia={slice_auc:.3f} | AUC paciente={p_auc:.3f} "
                  f"IC95%=[{lo:.3f}, {hi:.3f}]")
        print(f"  massa de prob. nos tokens ORIGINAIS (' Yes'/' No'): "
              f"mediana={g['mass_orig'].median():.2e}")
        print(f"  massa de prob. na família Yes/No:                  "
              f"mediana={g['mass_family'].median():.3f}")
        print(f"  d_family: média={g['d_family'].mean():+.2f}  std={g['d_family'].std():.2f}  "
              f"fração(Yes>No)={(g['d_family'] > 0).mean():.0%}")
        print(f"  top-1 mais frequentes: {g['top1'].value_counts().head(5).to_dict()}")
        by_pos = {int(i): round(auc_safe(gg['y_true'], gg['d_family']), 2)
                  for i, gg in g.groupby("slice_idx")}
        print(f"  AUC por posição da fatia (d_family): {by_pos}")

    g0 = df[df["variant"] == df["variant"].iloc[0]]
    print("\n--- confundidores de baixo nível (AUC por fatia, independe do modelo) ---")
    for col in ("img_mean", "img_fg_frac"):
        print(f"  {col:<12} AUC={auc_safe(g0['y_true'], g0[col]):.3f}")


@torch.inference_mode()
def describe_image(model, processor, image, max_new_tokens=90):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": DESCRIBE_PROMPT}, {"type": "image", "image": image}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=zs.DTYPE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def save_montage(patients, max_slices, path):
    fig, axes = plt.subplots(len(patients), max_slices, figsize=(1.6 * max_slices, 1.8 * len(patients)))
    axes = np.atleast_2d(axes)
    for r, rec in enumerate(patients):
        paths = zs.select_slices(rec["patient_dir"], max_slices)
        for c in range(max_slices):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(paths):
                ax.imshow(zs.load_slice_as_image(paths[c]))
            if c == 0:
                ax.set_title(f"{rec['label']} {rec['patient_id']}", fontsize=7, loc="left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_diagnostics(args):
    zs.seed_everything(args.seed)
    out_dir = zs.RESULTS_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = sample_val_patients(args.n_per_class, args.seed)
    model, processor = zs.load_model_and_processor(quantize_4bit=not args.no_4bit)
    tok = processor.tokenizer
    id_ad, id_cn = zs.get_answer_token_ids(processor)

    # --- auditoria dos tokens candidatos ---
    yes_map, no_map = token_ids(tok, YES_VARIANTS), token_ids(tok, NO_VARIANTS)
    print("\nTokens 'Yes' candidatos:", {k: (v, tok.convert_ids_to_tokens(v)) for k, v in yes_map.items()})
    print("Tokens 'No'  candidatos:", {k: (v, tok.convert_ids_to_tokens(v)) for k, v in no_map.items()})
    print(f"Tokens usados no script original: ' Yes' -> {id_ad} | ' No' -> {id_cn}")
    yes_ids = sorted(set(yes_map.values()))
    no_ids = sorted(set(no_map.values()))

    # --- como o template renderiza o prompt (onde começa a resposta?) ---
    first_img = zs.load_slice_as_image(zs.select_slices(patients[0]["patient_dir"], args.max_slices)[0])
    rendered = processor.apply_chat_template(
        build_messages("original", first_img), add_generation_prompt=True, tokenize=False)
    print("\nFINAL do prompt renderizado (a resposta começa logo depois):")
    print(repr(rendered[-120:]))

    # --- scoring por fatia, por variante ---
    rows = []
    for rec in tqdm(patients, desc="Pacientes"):
        paths = zs.select_slices(rec["patient_dir"], args.max_slices)
        for si, p in enumerate(paths):
            img = zs.load_slice_as_image(p)
            arr = np.asarray(img)[..., 0]
            stats = {"img_mean": float(arr.mean()), "img_fg_frac": float((arr > 10).mean())}
            for v in args.variants:
                logp = next_token_logprobs(model, processor, build_messages(v, img))
                r = score_row(logp, tok, yes_ids, no_ids, id_ad, id_cn)
                r.update(stats, patient_id=rec["patient_id"], variant=v, slice_idx=si,
                         y_true=0 if rec["label"] == zs.CLASS_1 else 1)
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "slice_scores.csv", index=False)
    summarize(df)

    # --- descrição em texto livre + imagens (o modelo "enxerga" um cérebro?) ---
    lines = []
    for cls in (zs.CLASS_1, zs.CLASS_2):
        for rec in [r for r in patients if r["label"] == cls][: args.describe]:
            paths = zs.select_slices(rec["patient_dir"], args.max_slices)
            img = zs.load_slice_as_image(paths[len(paths) // 2])
            lines.append(f"[{cls} {rec['patient_id']}] {describe_image(model, processor, img)}\n")
    (out_dir / "describe.txt").write_text("\n".join(lines))
    print("\n--- descrições livres (primeiras) ---")
    print("\n".join(lines[:4]))

    montage_patients = []
    for cls in (zs.CLASS_1, zs.CLASS_2):
        montage_patients += [r for r in patients if r["label"] == cls][:2]
    save_montage(montage_patients, args.max_slices, out_dir / "montage.png")
    print(f"\nSalvo em {out_dir}/ (slice_scores.csv, describe.txt, montage.png)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze-csv", nargs="+", help="só analisa predictions_test.csv existentes (sem GPU)")
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--max-slices", type=int, default=zs.MAX_SLICES_PER_PATIENT)
    ap.add_argument("--variants", nargs="+", default=["original", "image_first", "no_context", "minimal"],
                    choices=["original", "image_first", "no_context", "minimal"])
    ap.add_argument("--describe", type=int, default=2, help="pacientes por classe para descrição livre")
    ap.add_argument("--seed", type=int, default=zs.SEED)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if args.analyze_csv:
        analyze_csv(args.analyze_csv)
    else:
        run_diagnostics(args)


if __name__ == "__main__":
    main()