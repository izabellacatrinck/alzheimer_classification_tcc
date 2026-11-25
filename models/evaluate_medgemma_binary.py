"""
Avalia o MedGemma-4B-IT (zero-shot, pesos congelados) para classificação
binária CN vs AD, usando os mesmos pacientes e a mesma lógica de seleção/
pré-processamento de slices do DenseNet-121 (train_densenet_binary_cv.py) —
para permitir uma comparação válida entre as duas abordagens.

Como funciona:
  - Para cada slice selecionada, o prompt força o MedGemma a responder com
    UMA palavra: 'Yes' (Alzheimer's Disease) ou 'No' (Cognitively Normal) —
    formato validado na literatura (ver seção de formalização do prompt
    abaixo), não inventado.
  - Em vez de ler o texto gerado livremente, extraímos os LOGITS do
    primeiro (e único) token gerado (max_new_tokens=1, output_scores=True)
    e aplicamos softmax só nos dois tokens candidatos ('Yes' e 'No'). Isso dá
    uma probabilidade contínua e thresholdável — não um "sim/não" de texto
    livre — no mesmo espírito da calibração de threshold já usada com o
    DenseNet e o MedSigLIP (pesos sempre congelados, só o limiar de decisão
    é ajustado depois, fora deste script).
  - A predição do paciente é a MÉDIA das probabilidades das slices
    selecionadas (mesma agregação por paciente usada no DenseNet).
  - O tempo de inferência por paciente também é medido e salvo, no mesmo
    formato usado em measure_inference_time() do script do DenseNet, para
    a comparação de tempo do Sprint 2.

IMPORTANTE — o que este script NÃO faz: não calibra threshold nenhum. A
saída é só (patient_id, y_true, probability) bruto. Depois de rodar este
script, use calibrate_medgemma_threshold.py com o cv_fold_assignment.csv do
DenseNet para calibrar o threshold seguindo o mesmo protocolo cross-validado
(fold a fold) — só assim a comparação com compare_models.py fica livre de
vazamento.

Formalização do prompt (com base na literatura, não inventado): o formato de
pergunta binária forçada ("Answer with only Yes or No") seguido da leitura
dos logits do PRIMEIRO token gerado é o mesmo usado por Farina et al. (2026,
arXiv:2606.20852) para classificar pneumonia em radiografias de tórax com
esse MESMO checkpoint (MedGemma-4B-IT, pesos congelados) — adaptamos aqui
apenas o domínio (MRI cerebral) e a condição (Alzheimer's Disease). A
formalização matemática do escore (softmax entre os logits de "Yes" e "No",
equivalente a sigmoid da diferença dos logits) segue o mesmo procedimento
descrito por outro grupo independente (arXiv:2604.08941) para reduzir um
VLM generativo a um classificador binário apto a calibração de threshold —
exatamente o que fazemos depois, em calibrate_medgemma_threshold.py.
Um terceiro estudo (arXiv:2608.07550) usa essa mesma leitura de logits
Yes/No para medir confiança de VLMs médicos em radiografias, reforçando que
a abordagem é reconhecida na literatura recente, não uma escolha ad hoc.

LIMITAÇÃO A VALIDAR NO SEU AMBIENTE: o MedGemma é um modelo *gated* no
Hugging Face, e eu não tenho acesso a huggingface.co neste ambiente para
baixar os pesos e rodar isso de ponta a ponta antes de te entregar. Escrevi
com base na API oficial documentada (AutoModelForImageTextToText +
AutoProcessor.apply_chat_template) e no formato de prompt validado na
literatura acima, mas recomendo rodar primeiro com --limit 3 e conferir
manualmente:
  1) se os "token ids" impressos no início realmente correspondem a UM
     único token cada para 'Yes'/'No' (se não forem, o aviso no console vai
     dizer — os três estudos citados verificaram isso explicitamente para
     esse mesmo tokenizer antes de usar a técnica);
  2) se as probabilidades das 3 primeiras predições fazem sentido (não
     estão todas empacadas em ~0.5, o que indicaria que o modelo não está
     diferenciando as duas opções).

Requisitos:
    pip install transformers accelerate bitsandbytes torch pillow pandas numpy tqdm
    Acesso ao modelo google/medgemma-4b-it (gated — aceitar os termos de uso
    em huggingface.co/google/medgemma-4b-it e rodar `huggingface-cli login`
    antes de executar este script). GPU com pelo menos ~8-10 GB de VRAM
    (bf16) é recomendado.

Uso:
    # Teste rápido com 3 pacientes antes de rodar tudo:
    python evaluate_medgemma_binary.py \
        --fold-assignment results_densenet_cv_CN_vs_AD/cv_fold_assignment.csv \
        --limit 3

    # Rodada completa:
    python evaluate_medgemma_binary.py \
        --fold-assignment results_densenet_cv_CN_vs_AD/cv_fold_assignment.csv \
        --output-predictions medgemma_raw_predictions.csv \
        --output-inference-time medgemma_inference_time.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

MODEL_ID = "google/medgemma-4b-it"

CLASS_1 = "CN"  # Cognitively Normal -> resposta forçada "B"
CLASS_2 = "AD"  # Alzheimer's Disease -> resposta forçada "A"

# Mesma organização de pastas do train_densenet_binary_cv.py.
TRAIN_DIR = Path("Data/ADNI_FINAL_Updated/train")
TEST_DIR = Path("Data/ADNI_FINAL_Updated/test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MAX_SLICES_PER_PATIENT = 10  # mesma lógica de seleção central do DenseNet

# Tokens candidatos para a resposta forçada. Espaço à frente é proposital —
# em tokenizers estilo SentencePiece/BPE (caso do Gemma), uma palavra logo
# após um espaço costuma ser um token diferente da mesma palavra no início
# da string, e é assim que ela aparece depois do prompt (que termina com
# espaço antes da resposta). "Yes"/"No" (não "A"/"B") porque é o par de
# tokens já validado como resposta de 1 único token para o MedGemma-4B-IT
# na literatura citada no cabeçalho do arquivo.
ANSWER_TOKEN_AD = " Yes"  # AD (classe positiva, label=1)
ANSWER_TOKEN_CN = " No"   # CN (classe negativa, label=0)

# Prompt v2 — "rubric-anchored", não mais "minimal". A estrutura da pergunta
# forçada ("Answer with only Yes or No") vem de Farina et al. (2026,
# arXiv:2606.20852), que usa exatamente esse formato com o mesmo checkpoint
# para classificar pneumonia em raio-X. A ÂNCORA DE CRITÉRIOS (nomear os
# achados radiológicos específicos a procurar) foi adicionada depois de
# encontrar evidência de que prompts "minimal" (sem esses critérios) são
# especialmente propensos a "class collapse" em VLMs zero-shot — documentado
# formalmente por Goyal et al. (2026, arXiv:2606.21861), que mediram
# variações de até 32 pontos percentuais de acurácia dependendo só da
# formulação do prompt, com prompts "rubric-anchored" superando
# consistentemente prompts "minimal" equivalentes ao que eu tinha escrito
# antes. Os três achados estruturais citados (atrofia hipocampal, aumento
# ventricular, afinamento cortical) são os marcadores radiológicos
# estabelecidos de Alzheimer's na literatura de neurorradiologia.
PROMPT_TEXT = (
    "You are an expert neuroradiologist. Look at this axial brain MRI "
    "slice carefully. Established structural markers of Alzheimer's "
    "disease on MRI include hippocampal atrophy, ventricular enlargement, "
    "and cortical thinning, particularly in the medial temporal lobe. "
    "Considering these features, does this slice show signs of "
    "Alzheimer's disease? Answer with only Yes or No."
)

# Prompt v2 em duas etapas, para uso com --chain-of-thought (ver
# score_slice_cot). Baseado no framework de "thought chain" usado por um
# estudo dedicado à detecção de Alzheimer's com LLMs (Interpretable
# Alzheimer's Disease Detection with Minimal Data: Zero-Shot and Few-Shot
# Approaches Using LLMs, Springer 2025/2026), que obteve resultados mais
# precisos E mais interpretáveis pedindo uma análise passo a passo antes da
# resposta final, em vez de ir direto para Sim/Não.
COT_DESCRIBE_PROMPT = (
    "You are an expert neuroradiologist. Look at this axial brain MRI "
    "slice carefully. In one or two sentences, describe the ventricle "
    "size, the degree of cortical atrophy, and the appearance of the "
    "medial temporal lobe/hippocampal region."
)
COT_ANSWER_PROMPT = (
    "Based on this analysis, does this slice show signs of Alzheimer's "
    "disease? Answer with only Yes or No."
)

N_INFERENCE_WARMUP = 3
N_INFERENCE_MEASURE = 30


# ============================================================================
# CARREGAMENTO DE IMAGENS (mesma lógica do MRIDataset do DenseNet)
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
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def find_patient_dir(patient_id, label):
    for split_dir in (TRAIN_DIR, TEST_DIR):
        candidate = split_dir / label / patient_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Não encontrei a pasta do paciente {patient_id} (label={label}) em {TRAIN_DIR} nem {TEST_DIR}"
    )


# ============================================================================
# MODELO / TOKENS CANDIDATOS
# ============================================================================

def load_model_and_processor(quantize_4bit=True):
    print(f"Carregando {MODEL_ID} (isso pode demorar na primeira vez) ...")
    if quantize_4bit:
        # Essencial para GPUs com pouca VRAM (ex.: 6GB): em bf16 o modelo
        # sozinho já precisa de ~8GB só para os pesos, o que força
        # offloading para CPU/disco (o aviso "meta device" e a lentidão de
        # 30-50s/paciente vêm daí). Em 4-bit (NF4), os pesos caem para
        # ~2-2.5GB, cabendo inteiro na GPU mesmo com pouca VRAM.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, quantization_config=quant_config, device_map="auto",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def get_answer_token_ids(processor):
    tok_ad = processor.tokenizer.encode(ANSWER_TOKEN_AD, add_special_tokens=False)
    tok_cn = processor.tokenizer.encode(ANSWER_TOKEN_CN, add_special_tokens=False)
    if len(tok_ad) != 1 or len(tok_cn) != 1:
        print(
            f"AVISO: os tokens candidatos não correspondem a UM único token "
            f"(AD={ANSWER_TOKEN_AD!r} -> {tok_ad}, CN={ANSWER_TOKEN_CN!r} -> {tok_cn}). "
            f"Usando só o primeiro sub-token de cada um — confira manualmente com "
            f"--limit 3 se isso ainda é razoável para o tokenizer instalado."
        )
    id_ad, id_cn = tok_ad[0], tok_cn[0]
    # Confirma que os ids realmente decodificam de volta para "Yes"/"No" —
    # sanity check barato que evita concluir "o modelo tem viés" quando na
    # verdade o id capturado corresponde a outro token qualquer.
    decoded_ad = processor.tokenizer.decode([id_ad])
    decoded_cn = processor.tokenizer.decode([id_cn])
    print(f"Confirmação de decodificação -> id {id_ad} decodifica para {decoded_ad!r} "
          f"(esperado algo como 'Yes'); id {id_cn} decodifica para {decoded_cn!r} "
          f"(esperado algo como 'No').")
    if "yes" not in decoded_ad.lower() or "no" not in decoded_cn.lower():
        print("AVISO: a decodificação não bate com o esperado — os ids capturados "
              "provavelmente NÃO são os tokens 'Yes'/'No' de verdade. Isso invalidaria "
              "as probabilidades calculadas.")
    return id_ad, id_cn


# ============================================================================
# SCORING DE UMA SLICE / AGREGAÇÃO POR PACIENTE
# ============================================================================

@torch.inference_mode()
def score_slice(model, processor, image, id_ad, id_cn, verbose=False):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=DTYPE)

    outputs = model.generate(
        **inputs, max_new_tokens=1, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )
    logits = outputs.scores[0][0]  # primeiro (único) token gerado, batch_size=1
    logit_ad = logits[id_ad].item()
    logit_cn = logits[id_cn].item()
    two_class_logits = torch.tensor([logit_ad, logit_cn])
    prob_ad = torch.softmax(two_class_logits, dim=0)[0].item()

    if verbose:
        # argmax real do modelo nesse token — mostra o que ele "queria" dizer
        # de fato, não só a probabilidade forçada entre Yes/No.
        top_id = int(torch.argmax(logits).item())
        top_token = processor.tokenizer.decode([top_id])
        print(f"      [debug] logit(Yes)={logit_ad:+.3f}  logit(No)={logit_cn:+.3f}  "
              f"P(AD)={prob_ad:.4f}  |  token mais provável de todos: {top_token!r} (logit={logits[top_id].item():+.3f})")

    return prob_ad


@torch.inference_mode()
def debug_generate_free_text(model, processor, image, max_new_tokens=60):
    """Gera texto livre (sem forçar 1 token) para inspecionar o que o modelo
    realmente responderia sem a restrição Yes/No — útil pra distinguir
    'o modelo genuinamente diz não' de 'o modelo está recusando/hedging'."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=DTYPE)
    input_len = inputs["input_ids"].shape[-1]
    generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_tokens = generation[0][input_len:]
    return processor.decode(generated_tokens, skip_special_tokens=True)


@torch.inference_mode()
def score_slice_cot(model, processor, image, id_ad, id_cn, max_description_tokens=80, verbose=False):
    """
    Versão em duas etapas (chain-of-thought), para usar com --chain-of-thought.
    Mais cara (2 generate() por slice em vez de 1), mas baseada em evidência
    de que pedir uma análise passo a passo ANTES da resposta forçada reduz
    o colapso de classe em tarefas de detecção de Alzheimer's com LLMs (ver
    COT_DESCRIBE_PROMPT/COT_ANSWER_PROMPT no topo do arquivo para a fonte).
    """
    # Etapa 1: descrição livre dos achados (sem restrição de token)
    messages_stage1 = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": COT_DESCRIBE_PROMPT},
        ]}
    ]
    inputs1 = processor.apply_chat_template(
        messages_stage1, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=DTYPE)
    input_len1 = inputs1["input_ids"].shape[-1]
    gen1 = model.generate(**inputs1, max_new_tokens=max_description_tokens, do_sample=False)
    description = processor.decode(gen1[0][input_len1:], skip_special_tokens=True)

    if verbose:
        print(f"      [debug-cot] descrição gerada: {description}")

    # Etapa 2: resposta forçada (1 token), condicionada na descrição gerada
    messages_stage2 = messages_stage1 + [
        {"role": "assistant", "content": [{"type": "text", "text": description}]},
        {"role": "user", "content": [{"type": "text", "text": COT_ANSWER_PROMPT}]},
    ]
    inputs2 = processor.apply_chat_template(
        messages_stage2, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=DTYPE)
    outputs2 = model.generate(
        **inputs2, max_new_tokens=1, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )
    logits = outputs2.scores[0][0]
    logit_ad = logits[id_ad].item()
    logit_cn = logits[id_cn].item()
    prob_ad = torch.softmax(torch.tensor([logit_ad, logit_cn]), dim=0)[0].item()

    if verbose:
        print(f"      [debug-cot] logit(Yes)={logit_ad:+.3f}  logit(No)={logit_cn:+.3f}  P(AD)={prob_ad:.4f}")

    return prob_ad


def evaluate_patient(model, processor, patient_dir, id_ad, id_cn, max_slices=MAX_SLICES_PER_PATIENT,
                      verbose=False, use_cot=False):
    slice_paths = select_slices(patient_dir, max_slices=max_slices)
    scorer = score_slice_cot if use_cot else score_slice
    slice_probs = [
        scorer(model, processor, load_slice_as_image(p), id_ad, id_cn, verbose=verbose)
        for p in slice_paths
    ]
    return float(np.mean(slice_probs)), len(slice_probs)


# ============================================================================
# TEMPO DE INFERÊNCIA (mesmo formato usado no script do DenseNet)
# ============================================================================

def measure_inference_time(model, processor, patient_dirs, id_ad, id_cn,
                            n_warmup=N_INFERENCE_WARMUP, n_measure=N_INFERENCE_MEASURE,
                            max_slices=MAX_SLICES_PER_PATIENT, use_cot=False):
    sample = patient_dirs[: n_warmup + n_measure]
    warmup_dirs = sample[:n_warmup]
    measure_dirs = sample[n_warmup:]

    for pdir in warmup_dirs:
        evaluate_patient(model, processor, pdir, id_ad, id_cn, max_slices=max_slices, use_cot=use_cot)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times, n_slices_list = [], []
    for pdir in measure_dirs:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, n_slices = evaluate_patient(model, processor, pdir, id_ad, id_cn, max_slices=max_slices, use_cot=use_cot)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        n_slices_list.append(n_slices)

    times = np.array(times)
    return {
        "device": str(DEVICE),
        "model_id": MODEL_ID,
        "chain_of_thought": use_cot,
        "n_patients_measured": int(len(times)),
        "mean_slices_per_patient": float(np.mean(n_slices_list)) if n_slices_list else 0.0,
        "mean_seconds_per_patient": float(times.mean()) if len(times) else 0.0,
        "std_seconds_per_patient": float(times.std()) if len(times) else 0.0,
        "min_seconds_per_patient": float(times.min()) if len(times) else 0.0,
        "max_seconds_per_patient": float(times.max()) if len(times) else 0.0,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold-assignment", required=True,
                         help="cv_fold_assignment.csv gerado pelo train_densenet_binary_cv.py")
    parser.add_argument("--max-slices", type=int, default=MAX_SLICES_PER_PATIENT)
    parser.add_argument("--output-predictions", default="medgemma_raw_predictions.csv")
    parser.add_argument("--output-inference-time", default="medgemma_inference_time.json")
    parser.add_argument("--limit", type=int, default=None,
                         help="avaliar só os N primeiros pacientes (teste rápido, ex.: --limit 3)")
    parser.add_argument("--patient-ids", default=None,
                         help="lista de patient_id separados por vírgula, para testar pacientes "
                              "específicos (ex.: um CN e um AD) em vez dos N primeiros do CSV — "
                              "sobrepõe --limit se os dois forem passados")
    parser.add_argument("--debug", action="store_true",
                         help="imprime logits brutos de Yes/No por slice, o token mais provável "
                              "segundo o modelo, e gera texto livre (sem forçar 1 token) para a "
                              "primeira slice do primeiro paciente, pra diagnosticar respostas "
                              "estranhas (ex.: probabilidade sempre 0.0000)")
    parser.add_argument("--chain-of-thought", action="store_true",
                         help="usa o prompt em duas etapas (descrição livre + resposta forçada) "
                              "em vez da resposta direta — mais caro (2 generate() por slice), "
                              "mas ajuda a evitar colapso de classe (ver COT_DESCRIBE_PROMPT no "
                              "topo do arquivo para a fonte). Tente primeiro sem essa flag; "
                              "escale para ela só se o colapso persistir.")
    parser.add_argument("--no-4bit", action="store_true",
                         help="desativa a quantização 4-bit e carrega em bf16/fp32 — só use se "
                              "tiver VRAM de sobra (~10GB+); em GPUs pequenas (6-8GB) deixe a "
                              "quantização 4-bit ligada (padrão) para evitar offloading para CPU/disco")
    parser.add_argument("--skip-inference-timing", action="store_true",
                         help="pula a medição de tempo de inferência (só gera as predições)")
    args = parser.parse_args()

    fold_df = pd.read_csv(args.fold_assignment)
    if args.patient_ids:
        wanted = [pid.strip() for pid in args.patient_ids.split(",")]
        fold_df = fold_df.set_index("patient_id").loc[wanted].reset_index()
    elif args.limit:
        fold_df = fold_df.iloc[: args.limit]
    print(f"Pacientes a avaliar: {len(fold_df)}")
    if args.chain_of_thought:
        print("Modo chain-of-thought ATIVADO (2 generate() por slice — mais lento).")

    label_map = {CLASS_1: 0, CLASS_2: 1}

    model, processor = load_model_and_processor(quantize_4bit=not args.no_4bit)
    id_ad, id_cn = get_answer_token_ids(processor)
    print(f"Token id resposta 'Yes' (AD): {id_ad} | Token id resposta 'No' (CN): {id_cn}")

    rows = []
    patient_dirs_seen = []
    verbose_mode = args.debug or bool(args.limit) or bool(args.patient_ids)

    for i, (_, row) in enumerate(tqdm(fold_df.iterrows(), total=len(fold_df), desc="Avaliando pacientes")):
        patient_id = row["patient_id"]
        label = row["label"]
        patient_dir = find_patient_dir(patient_id, label)
        patient_dirs_seen.append(patient_dir)

        if args.debug and i == 0 and not args.chain_of_thought:
            first_slice = select_slices(patient_dir, max_slices=args.max_slices)[0]
            free_text = debug_generate_free_text(model, processor, load_slice_as_image(first_slice))
            print(f"\n  [debug] Geração livre (sem forçar 1 token) para {patient_id}, "
                  f"primeira slice:\n  ---\n  {free_text}\n  ---\n")

        prob_ad, n_slices = evaluate_patient(model, processor, patient_dir, id_ad, id_cn,
                                              max_slices=args.max_slices, verbose=args.debug,
                                              use_cot=args.chain_of_thought)

        rows.append({
            "patient_id": patient_id,
            "y_true": label_map[label],
            "probability": prob_ad,
            "n_slices": n_slices,
        })

        if verbose_mode:
            print(f"  {patient_id} (label={label}) -> P(AD)={prob_ad:.4f} ({n_slices} slices)")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(args.output_predictions, index=False)
    print(f"\nPrevisões salvas em: {args.output_predictions}")

    if not args.skip_inference_timing and len(patient_dirs_seen) > N_INFERENCE_WARMUP:
        print("\n===== Medindo tempo de inferência (MedGemma) =====")
        inference_stats = measure_inference_time(model, processor, patient_dirs_seen, id_ad, id_cn,
                                                   max_slices=args.max_slices, use_cot=args.chain_of_thought)
        print(json.dumps(inference_stats, indent=2))
        with open(args.output_inference_time, "w") as f:
            json.dump(inference_stats, f, indent=4)
        print(f"Tempo de inferência salvo em: {args.output_inference_time}")


if __name__ == "__main__":
    main()