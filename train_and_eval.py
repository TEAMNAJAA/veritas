import os
import re
import json
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset as TorchDataset
from safetensors.torch import save_file
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import jiwer
from tqdm.auto import tqdm
from accent_engine import AccentBridgeEngine, STOPWORDS, normalize_text, phonetic_similarity

random.seed(42)
np.random.seed(42)

NATIVE_WESTERN = {
    "mainstream us english", "southern british english", "scottish english",
    "irish english", "canadian english", "australian english", "new zealand english"
}
EAST_SE_ASIAN = ["vietnam", "mandarin", "chinese", "cantonese", "indonesia", "thai", "korea", "japan", "malay", "tagalog", "filipino"]
SOUTH_ASIAN = ["hindi", "indian", "tamil", "urdu", "bengali", "punjabi", "telugu"]

def categorize_accent(l1_str: str, accent_str: str) -> str:
    comb = f"{l1_str} {accent_str}".lower()
    if any(k in comb for k in EAST_SE_ASIAN):
        return "East/SE Asian"
    if any(k in comb for k in SOUTH_ASIAN):
        return "South Asian"
    return "Other L2 Accents"

def is_l2_hard_phonology_word(w: str) -> bool:
    if len(w) < 4 or w in STOPWORDS:
        return False
    return bool(re.search(r"(th|ght|str|spr|[rl]|(ld|st|nd|nt|ct|pt|ft|le|ce|se))$", w))

def clean_transcript(text):
    if not text:
        return None
    if any(tag in text.upper() for tag in ["<FOREIGN>", "<INAUDIBLE>", "<OVERLAP>", "<UNINTELLIGIBLE>"]):
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text.split()) >= 4 else None

def collect_edacc_samples(stream, max_samples):
    rows = []
    for item in stream:
        l1 = str(item.get("l1", "")).strip()
        accent = str(item.get("accent", "")).strip()
        if l1.lower() in NATIVE_WESTERN:
            continue
        clean_txt = clean_transcript(item.get("text", ""))
        if not clean_txt:
            continue
        arr = item["audio"]["array"]
        dur = len(arr) / 16000.0
        if not (2.0 <= dur <= 18.0):
            continue
        rows.append({"audio_array": arr, "text": clean_txt, "l1": l1, "group": categorize_accent(l1, accent)})
        if len(rows) >= max_samples:
            break
    return rows

class AccentTorchDataset(TorchDataset):
    def __init__(self, items, processor):
        self.items, self.processor = items, processor
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        row = self.items[idx]
        feat = self.processor.feature_extractor(row["audio_array"], sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(row["text"], truncation=True, max_length=128).input_ids
        return {"input_features": feat, "labels": labels}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    bos_token_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        feats = torch.stack([torch.as_tensor(f["input_features"], dtype=torch.float32) for f in features])
        label_tensors = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]
        padded_labels = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        if padded_labels.size(1) > 0 and (padded_labels[:, 0] == self.bos_token_id).all().item():
            padded_labels = padded_labels[:, 1:]
        return {"input_features": feats, "labels": padded_labels}

def mine_confusion_priors(model, processor, samples, device, out_json_path: str):
    model.eval()
    pair_counts = Counter()
    for r in tqdm(samples[:180], desc="Mining L2 Phonetic Confusion Priors"):
        feats = processor(r["audio_array"], sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=128)
        hyp_words = normalize_text(processor.batch_decode(ids, skip_special_tokens=True)[0]).split()
        ref_words = r["text"].split()
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, hyp_words, ref_words).get_opcodes():
            if tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
                hw, rw = hyp_words[i1], ref_words[j1]
                if hw != rw and len(rw) >= 3 and rw not in STOPWORDS and phonetic_similarity(hw, rw) >= 0.55:
                    pair_counts[f"{hw}->{rw}"] += 1
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"word_pairs": {k: v for k, v in pair_counts.most_common(250)}}, f, indent=2)

def calibrate_hyperparameters_on_dev(engine: AccentBridgeEngine, dev_samples: list, out_path: str) -> dict:
    dev_slice = dev_samples[:30]
    dev_pool = list({w for r in dev_slice for w in r["text"].split() if len(w) >= 5 and w not in STOPWORDS})
    candidates = [
        {"alpha": 0.12, "hw_weight": 0.16, "gate_threshold": 0.85},
        {"alpha": 0.16, "hw_weight": 0.20, "gate_threshold": 0.88},
        {"alpha": 0.22, "hw_weight": 0.24, "gate_threshold": 0.90},
    ]
    best_cfg, best_score = candidates[1], 999.0
    for cfg in candidates:
        refs, preds, false_edits = [], [], 0
        for r in dev_slice:
            ref = r["text"]
            refs.append(ref)
            kws = [w for w in dict.fromkeys(ref.split()) if len(w) >= 5 and w not in STOPWORDS][:3]
            dist = random.sample([w for w in dev_pool if w not in set(ref.split())], k=min(15, max(0, len(dev_pool) - 4)))
            p_txt, _ = engine.transcribe_accentbridge(
                r["audio_array"], hotwords_raw=kws + dist,
                alpha=cfg["alpha"], hw_weight=cfg["hw_weight"], gate_thresh=cfg["gate_threshold"],
            )
            _, wrong_r = engine.transcribe_accentbridge(
                r["audio_array"], hotwords_raw=dist,
                alpha=cfg["alpha"], hw_weight=cfg["hw_weight"], gate_thresh=cfg["gate_threshold"],
            )
            preds.append(p_txt)
            false_edits += len(wrong_r)
        composite = (jiwer.wer(refs, preds) * 100.0) + (0.25 * false_edits)
        if composite < best_score:
            best_score, best_cfg = composite, cfg
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)
    engine.calib.update(best_cfg)
    return best_cfg

def quick_wer(model, processor, samples, device):
    model.eval()
    refs, preds = [], []
    for r in samples:
        feats = processor(r["audio_array"], sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=128)
        preds.append(normalize_text(processor.batch_decode(ids, skip_special_tokens=True)[0]))
        refs.append(r["text"])
    return jiwer.wer(refs, preds) * 100.0

def keyword_recall(target_words_list, predictions):
    total_kw, hit_kw = 0, 0
    for kws, pred in zip(target_words_list, predictions):
        pred_set = set(pred.split())
        for kw in kws:
            total_kw += 1
            if kw in pred_set:
                hit_kw += 1
    return (hit_kw / max(1, total_kw)) * 100.0

def paired_bootstrap_ci(refs: list, base_preds: list, ab_preds: list, n_boot: int = 1000):
    n = len(refs)
    lengths = np.array([max(1, len(r.split())) for r in refs], dtype=np.float64)
    base_errs = np.array([jiwer.wer(r, b) * l for r, b, l in zip(refs, base_preds, lengths)])
    ab_errs = np.array([jiwer.wer(r, p) * l for r, p, l in zip(refs, ab_preds, lengths)])
    diffs, wins = [], 0
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        tot_len = lengths[idx].sum()
        b_w, a_w = (base_errs[idx].sum() / tot_len) * 100.0, (ab_errs[idx].sum() / tot_len) * 100.0
        diffs.append(a_w - b_w)
        if a_w < b_w:
            wins += 1
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(ci_low), float(ci_high), float(1.0 - (wins / n_boot))

def save_charts(wer_dict: dict, kw_dict: dict, group_wers: dict, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    bars1 = ax1.bar(list(wer_dict.keys()), list(wer_dict.values()), color=["#94a3b8", "#60a5fa", "#16a34a"])
    ax1.set_title("Overall Word Error Rate (Lower is Better)", fontweight="bold")
    ax1.set_ylabel("WER (%)")
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.2f}%", ha="center", fontweight="bold")

    bars2 = ax2.bar(list(kw_dict.keys()), list(kw_dict.values()), color=["#94a3b8", "#3b82f6", "#15803d"])
    ax2.set_title("Slide Keyword Recall (Higher is Better)", fontweight="bold")
    ax2.set_ylabel("Recall (%)")
    ax2.set_ylim(0, min(105, max(kw_dict.values()) + 15))
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.2f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_chart.png"))
    plt.close()

    if group_wers:
        groups = list(group_wers.keys())
        x, w = np.arange(len(groups)), 0.35
        fig2, ax = plt.subplots(figsize=(8.5, 4.5), dpi=200)
        r1 = ax.bar(x - w / 2, [group_wers[g][0] for g in groups], w, label="Baseline Whisper", color="#94a3b8")
        r2 = ax.bar(x + w / 2, [group_wers[g][1] for g in groups], w, label="AccentBridge", color="#16a34a")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontweight="bold")
        ax.set_ylabel("WER (%)")
        ax.set_title("WER Reduction Breakdown by Speaker Accent Region", fontweight="bold")
        ax.legend()
        for r in list(r1) + list(r2):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.4, f"{r.get_height():.1f}%", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_accent_wer.png"))
        plt.close()

def main():
    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"
    train_size, dev_size = (420, 60) if has_gpu else (120, 30)
    max_steps, test_size = (260 if has_gpu else 60), 130

    edacc = load_dataset("edinburghcstr/edacc", streaming=True)
    val_all = collect_edacc_samples(edacc["validation"].cast_column("audio", Audio(sampling_rate=16000)), train_size + dev_size)
    train_data, dev_data = val_all[:train_size], val_all[train_size:]
    test_data = collect_edacc_samples(edacc["test"].cast_column("audio", Audio(sampling_rate=16000)), test_size)

    model_id = "openai/whisper-tiny.en"
    processor = WhisperProcessor.from_pretrained(model_id)
    ft_model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

    os.makedirs("./saved_model", exist_ok=True)
    mine_confusion_priors(ft_model, processor, train_data, device, "./saved_model/confusion_priors.json")
    dev_base_wer = quick_wer(ft_model, processor, dev_data, device)

    ft_model.config.use_cache = False
    ft_model.config.apply_spec_augment = True
    ft_model.config.mask_time_prob = 0.05
    ft_model.config.mask_feature_prob = 0.04

    # ปลดล็อกเฉพาะ Attention Projections ของเลเยอร์บนและ Decoder (~5.6M พารามิเตอร์ = ไฟล์เซฟขนาดเพียง ~11 MB)
    trainable_keys = []
    for name, param in ft_model.named_parameters():
        is_attn_proj = any(proj in name for proj in ("q_proj", "k_proj", "v_proj", "out_proj"))
        is_early_enc = any(prefix in name for prefix in ("encoder.layers.0.", "encoder.layers.1."))
        if is_attn_proj and not is_early_enc:
            param.requires_grad = True
            trainable_keys.append(name)
        else:
            param.requires_grad = False

    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            output_dir="./accentbridge_ckpt",
            per_device_train_batch_size=8 if has_gpu else 4,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            warmup_steps=30,
            max_steps=max_steps,
            fp16=has_gpu,
            logging_steps=40,
            save_strategy="no",
            report_to="none",
        ),
        model=ft_model,
        train_dataset=AccentTorchDataset(train_data, processor),
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(bos_token_id=processor.tokenizer.bos_token_id),
    )
    trainer.train()

    dev_ft_wer = quick_wer(ft_model.to(device), processor, dev_data, device)
    if dev_ft_wer <= dev_base_wer + 0.5:
        full_sd = ft_model.state_dict()
        delta_fp16 = {k: full_sd[k].detach().half().cpu() for k in trainable_keys}
        save_file(delta_fp16, "./saved_model/accent_attn.safetensors")
        print(f"✅ บันทึกน้ำหนักโมเดลขนาดกะทัดรัด (~11 MB) ลง ./saved_model/accent_attn.safetensors สำเร็จ!")

    engine = AccentBridgeEngine(finetuned_dir="./saved_model", base_id=model_id)
    best_cfg = calibrate_hyperparameters_on_dev(engine, dev_data, "./saved_model/calibrated_params.json")

    vocab_pool = list({w for r in test_data for w in r["text"].split() if len(w) >= 5 and w not in STOPWORDS})
    refs, base_preds, ft_only_preds, full_preds = [], [], [], []
    true_keywords_per_row, hard_phon_per_row = [], []
    wrong_slide_ungated, wrong_slide_gated = 0, 0
    group_indices = defaultdict(list)

    os.makedirs("./demo_samples", exist_ok=True)
    saved_demo_clips = []

    for idx, row in enumerate(tqdm(test_data, desc="Evaluating Final Benchmark")):
        ref = row["text"]
        refs.append(ref)
        group_indices[row["group"]].append(idx)

        true_kws = [w for w in dict.fromkeys(ref.split()) if len(w) >= 5 and w not in STOPWORDS][:3]
        hard_kws = [w for w in dict.fromkeys(ref.split()) if is_l2_hard_phonology_word(w)][:4]
        true_keywords_per_row.append(true_kws)
        hard_phon_per_row.append(hard_kws)

        distractors = random.sample([w for w in vocab_pool if w not in set(ref.split())], k=min(20, max(0, len(vocab_pool) - 5)))
        slide_vocab = list(dict.fromkeys(true_kws + hard_kws + distractors))
        random.shuffle(slide_vocab)

        b_txt = engine.transcribe_baseline(row["audio_array"])
        ft_txt, _ = engine.transcribe_accentbridge(row["audio_array"], hotwords_raw=None)
        ab_txt, _ = engine.transcribe_accentbridge(row["audio_array"], hotwords_raw=slide_vocab, slide_topic=" ".join(true_kws), use_confidence_gate=True)

        base_preds.append(b_txt)
        ft_only_preds.append(ft_txt)
        full_preds.append(ab_txt)

        if len(saved_demo_clips) < 2 and jiwer.wer(ref, ab_txt) < jiwer.wer(ref, b_txt) and true_kws:
            wav_path = f"./demo_samples/sample_{len(saved_demo_clips) + 1}.wav"
            sf.write(wav_path, row["audio_array"], 16000)
            saved_demo_clips.append({
                "file": wav_path,
                "l1": row["l1"],
                "keywords": ", ".join(list(dict.fromkeys(true_kws + hard_kws))),
                "ground_truth": ref,
            })

        _, ungated_r = engine.transcribe_accentbridge(row["audio_array"], hotwords_raw=distractors, use_confidence_gate=False)
        _, gated_r = engine.transcribe_accentbridge(row["audio_array"], hotwords_raw=distractors, use_confidence_gate=True)
        wrong_slide_ungated += len(ungated_r)
        wrong_slide_gated += len(gated_r)

    with open("./demo_samples/manifest.json", "w", encoding="utf-8") as f:
        json.dump(saved_demo_clips, f, indent=2)

    base_wer, ft_wer, full_wer = jiwer.wer(refs, base_preds) * 100, jiwer.wer(refs, ft_only_preds) * 100, jiwer.wer(refs, full_preds) * 100
    base_kw, ft_kw, full_kw = keyword_recall(true_keywords_per_row, base_preds), keyword_recall(true_keywords_per_row, ft_only_preds), keyword_recall(true_keywords_per_row, full_preds)
    base_hard, full_hard = keyword_recall(hard_phon_per_row, base_preds), keyword_recall(hard_phon_per_row, full_preds)
    mean_diff, ci_low, ci_high, p_val = paired_bootstrap_ci(refs, base_preds, full_preds)

    group_wers = {
        g: (jiwer.wer([refs[i] for i in idxs], [base_preds[i] for i in idxs]) * 100, jiwer.wer([refs[i] for i in idxs], [full_preds[i] for i in idxs]) * 100)
        for g, idxs in group_indices.items() if len(idxs) >= 5
    }

    os.makedirs("./results", exist_ok=True)
    save_charts({"Baseline": base_wer, "Fine-Tuned": ft_wer, "AccentBridge": full_wer}, {"Baseline": base_kw, "Fine-Tuned": ft_kw, "AccentBridge": full_kw}, group_wers, "./results")
    red_pct = ((wrong_slide_ungated - wrong_slide_gated) / max(1, wrong_slide_ungated)) * 100.0 if wrong_slide_ungated > 0 else 100.0

    lines = [
        "# AccentBridge — Evaluation Report (EdAcc Unseen Speakers)\n",
        "## 1. Main Ablation Benchmark",
        "| Model Variant | Overall WER (Lower=Better) | Slide Keyword Recall | L2 Hard-Phonology Recall (th, r/l, clusters) |",
        "|---|---|---|---|",
        f"| 1. Baseline (Zero-shot whisper-tiny.en) | {base_wer:.2f}% | {base_kw:.2f}% | {base_hard:.2f}% |",
        f"| 2. Accent-Adapted Whisper (Attention Delta 11MB) | {ft_wer:.2f}% | {ft_kw:.2f}% | — |",
        f"| 3. AccentBridge Full (+CMU ARPAbet Phonemes & Dev-Calibrated Gate) | {full_wer:.2f}% | {full_kw:.2f}% | {full_hard:.2f}% |\n",
        f"- **Dev-Calibrated Parameters:** `alpha={best_cfg['alpha']}`, `hw_weight={best_cfg['hw_weight']}`, `gate={best_cfg['gate_threshold']}`",
        f"- **Paired Bootstrap 95% CI:** `{mean_diff:+.2f}%` (`[{ci_low:+.2f}%, {ci_high:+.2f}%]`, **p = {p_val:.4f}**)",
        f"- **Wrong-Slide False-Edit Reduction:** `{wrong_slide_ungated}` → `{wrong_slide_gated}` (**{red_pct:.1f}% safer**)\n",
        "## 2. Winning Examples & Failure Cases",
    ]
    for r, b, p, meta in list(zip(refs, base_preds, full_preds, test_data))[:60]:
        if jiwer.wer(r, p) < jiwer.wer(r, b):
            lines.append(f"- **[WIN - {meta['l1']}]** Truth: `{r}` | Baseline: `{b}` | AccentBridge: `{p}`")
    for r, p, meta in list(zip(refs, full_preds, test_data))[:60]:
        if jiwer.wer(r, p) > 0.35:
            lines.append(f"- **[FAIL - {meta['l1']}]** Truth: `{r}` | Predicted: `{p}`")

    with open("./results/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines[:18]))

if __name__ == "__main__":
    main()
