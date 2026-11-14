import os
import re
import json
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from slide_parser import extract_priority_keywords, extract_context_bigrams, extract_acronyms

try:
    import pronouncing
    HAS_CMUDICT = True
except ImportError:
    HAS_CMUDICT = False

STOPWORDS = {
    "a", "an", "the", "to", "in", "on", "at", "of", "for", "and", "or",
    "is", "am", "are", "was", "were", "be", "it", "he", "she", "we", "they",
    "i", "you", "my", "me", "do", "go", "so", "no", "up", "by", "as", "not"
}

HALLUCINATION_PHRASES = {
    "thank you", "thank you for watching", "thanks for watching", "you", "bye", "subtitles by"
}

ARPABET_PAIR_COSTS = {
    frozenset({"R", "L"}): 0.12,
    frozenset({"R", "ER"}): 0.14,
    frozenset({"L", "ER"}): 0.18,
    frozenset({"TH", "T"}): 0.12,
    frozenset({"TH", "S"}): 0.18,
    frozenset({"DH", "D"}): 0.12,
    frozenset({"DH", "Z"}): 0.18,
    frozenset({"V", "W"}): 0.14,
    frozenset({"V", "F"}): 0.18,
    frozenset({"T", "D"}): 0.14,
    frozenset({"P", "B"}): 0.16,
    frozenset({"K", "G"}): 0.16,
    frozenset({"S", "Z"}): 0.14,
    frozenset({"SH", "CH"}): 0.16,
    frozenset({"SH", "S"}): 0.20,
    frozenset({"CH", "TS"}): 0.18,
    frozenset({"JH", "CH"}): 0.20,
    frozenset({"N", "NG"}): 0.20,
    frozenset({"IY", "IH"}): 0.16,
    frozenset({"UW", "UH"}): 0.16,
    frozenset({"EY", "EH"}): 0.20,
    frozenset({"AE", "EH"}): 0.20,
    frozenset({"AA", "AO"}): 0.16,
    frozenset({"OW", "AO"}): 0.20,
    frozenset({"AH", "ER"}): 0.20,
}
ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}
DROPPABLE_CODA_PHONEMES = {"T", "D", "K", "G", "P", "S", "Z", "L", "R"}

def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def fallback_grapheme_phones(word: str) -> list:
    w = re.sub(r"[^a-z]", "", word.lower())
    w = re.sub(r"^wr", "r", w)
    w = re.sub(r"^kn", "n", w)
    w = re.sub(r"tion$", "shn", w)
    w = w.replace("ght", "t").replace("ph", "f").replace("ck", "k")
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    tokens, i = [], 0
    while i < len(w):
        if w[i : i + 2] in ("th", "sh", "ch", "ng"):
            tokens.append(w[i : i + 2].upper())
            i += 2
        else:
            ch = w[i].upper()
            tokens.append({"A": "AE", "E": "EH", "I": "IH", "O": "AA", "U": "AH"}.get(ch, ch))
            i += 1
    return tokens or [word.upper()]

def word_to_arpabet(word: str) -> list:
    clean_w = word.lower().strip("'")
    if HAS_CMUDICT:
        phones_list = pronouncing.phones_for_word(clean_w)
        if not phones_list and len(clean_w) > 3 and clean_w.endswith("s"):
            base_phones = pronouncing.phones_for_word(clean_w[:-1])
            if base_phones:
                return [re.sub(r"\d", "", p) for p in base_phones[0].split()] + ["Z"]
        if phones_list:
            return [re.sub(r"\d", "", p) for p in phones_list[0].split()]
    return fallback_grapheme_phones(clean_w)

def _phoneme_sub_cost(p1: str, p2: str) -> float:
    if p1 == p2:
        return 0.0
    pair = frozenset({p1, p2})
    if pair in ARPABET_PAIR_COSTS:
        return ARPABET_PAIR_COSTS[pair]
    if p1 in ARPABET_VOWELS and p2 in ARPABET_VOWELS:
        return 0.34
    return 1.0

def phonetic_similarity(w1: str, w2: str) -> float:
    w1_c, w2_c = w1.lower().strip("'"), w2.lower().strip("'")
    if not w1_c or not w2_c:
        return 0.0
    if w1_c == w2_c:
        return 1.0

    ph1, ph2 = word_to_arpabet(w1_c), word_to_arpabet(w2_c)
    if ph1 == ph2:
        return 0.97

    m, n = len(ph1), len(ph2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + (0.20 if (i == m and ph1[i - 1] in DROPPABLE_CODA_PHONEMES) else 0.88)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + (0.20 if (j == n and ph2[j - 1] in DROPPABLE_CODA_PHONEMES) else 0.88)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            del_c = 0.20 if (i == m and ph1[i - 1] in DROPPABLE_CODA_PHONEMES) else 0.88
            ins_c = 0.20 if (j == n and ph2[j - 1] in DROPPABLE_CODA_PHONEMES) else 0.88
            sub_c = _phoneme_sub_cost(ph1[i - 1], ph2[j - 1])
            dp[i][j] = min(dp[i - 1][j] + del_c, dp[i][j - 1] + ins_c, dp[i - 1][j - 1] + sub_c)

    return float(max(0.0, 1.0 - (dp[m][n] / max(m, n, 1))))

def normalize_audio_safe(raw_arr: np.ndarray, orig_sr: int) -> np.ndarray:
    arr = np.asarray(raw_arr)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if orig_sr != 16000:
        arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=16000)
    if len(arr) == 0:
        return arr
    p99 = float(np.percentile(np.abs(arr), 99.2))
    if p99 > 1e-6:
        arr = np.clip(arr / p99, -1.0, 1.0)
    return arr

def preprocess_and_chunk_audio(audio_16k: np.ndarray, sr: int = 16000, max_chunk_sec: float = 24.0) -> list:
    if audio_16k is None or len(audio_16k) < int(0.35 * sr):
        return []
    trimmed, _ = librosa.effects.trim(audio_16k.astype(np.float32), top_db=28)
    rms = float(np.sqrt(np.mean(np.square(trimmed)))) if len(trimmed) > 0 else 0.0
    if len(trimmed) < int(0.35 * sr) or rms < 1e-4:
        return []

    max_len = int(max_chunk_sec * sr)
    if len(trimmed) <= max_len:
        return [trimmed]

    chunks, cursor = [], 0
    min_len, win = int(14.0 * sr), int(0.25 * sr)
    while cursor < len(trimmed):
        if len(trimmed) - cursor <= max_len:
            chunks.append(trimmed[cursor:])
            break
        search_zone = trimmed[cursor + min_len : cursor + max_len]
        best_offset, lowest_energy = max_len, 1e9
        for pos in range(0, max(1, len(search_zone) - win), win // 2):
            seg_energy = float(np.mean(np.square(search_zone[pos : pos + win])))
            if seg_energy < lowest_energy:
                lowest_energy, best_offset = seg_energy, min_len + pos + (win // 2)
        chunks.append(trimmed[cursor : cursor + best_offset])
        cursor += best_offset

    return [c for c in chunks if len(c) >= int(0.35 * sr)]


class AccentBridgeEngine:
    def __init__(self, finetuned_dir: str = "./saved_model", base_id: str = "openai/whisper-tiny.en"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.base_id = base_id
        self.processor = WhisperProcessor.from_pretrained(base_id)
        self.baseline_model = WhisperForConditionalGeneration.from_pretrained(base_id).to(self.device)
        self.baseline_model.eval()

        delta_file = os.path.join(finetuned_dir, "accent_attn.safetensors")
        if os.path.exists(delta_file):
            self.ft_model = WhisperForConditionalGeneration.from_pretrained(base_id)
            delta_state = {k: v.float() for k, v in load_file(delta_file).items()}
            self.ft_model.load_state_dict(delta_state, strict=False)
            self.ft_model = self.ft_model.to(self.device)
            self.using_finetuned = True
        elif os.path.isdir(finetuned_dir) and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(finetuned_dir)):
            self.ft_model = WhisperForConditionalGeneration.from_pretrained(finetuned_dir).to(self.device)
            self.using_finetuned = True
        else:
            self.ft_model = self.baseline_model
            self.using_finetuned = False
        self.ft_model.eval()

        self.learned_confusions = {}
        priors_path = os.path.join(finetuned_dir, "confusion_priors.json")
        if os.path.exists(priors_path):
            try:
                with open(priors_path, "r", encoding="utf-8") as f:
                    self.learned_confusions = json.load(f)
            except Exception:
                pass

        self.calib = {"alpha": 0.15, "hw_weight": 0.18, "gate_threshold": 0.88}
        calib_path = os.path.join(finetuned_dir, "calibrated_params.json")
        if os.path.exists(calib_path):
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    self.calib.update(json.load(f))
            except Exception:
                pass

        self.lm_tok = AutoTokenizer.from_pretrained("distilgpt2")
        if self.lm_tok.pad_token is None:
            self.lm_tok.pad_token = self.lm_tok.eos_token
        self.lm_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(self.device)
        self.lm_model.eval()
        self.last_word_confidences = {}

    def score_sentences_batched(self, texts: list, slide_topic: str = "") -> list:
        if not texts:
            return []
        topic_clean = normalize_text(slide_topic)[:50]
        prefix = f"topic {topic_clean}: " if topic_clean else ""
        prepared = [prefix + t.strip() if len(t.strip().split()) >= 2 else t.strip() for t in texts]
        non_empty = [p if p else "empty" for p in prepared]

        enc = self.lm_tok(non_empty, return_tensors="pt", padding=True, truncation=True, max_length=160).to(self.device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]
        if input_ids.shape[1] < 2:
            return [-8.0] * len(texts)

        with torch.no_grad():
            logits = self.lm_model(input_ids=input_ids, attention_mask=attn_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attn_mask[:, 1:].contiguous().float()
            tok_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            seq_loss = (tok_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1.0)

        scores = (-seq_loss).cpu().tolist()
        return [sc if len(t.strip().split()) >= 2 else -8.0 for sc, t in zip(scores, texts)]

    def parse_hotwords(self, hotwords_raw, context_text: str = "") -> list:
        combined = []
        if hotwords_raw:
            items = re.split(r"[,\n;]+", hotwords_raw) if isinstance(hotwords_raw, str) else list(hotwords_raw)
            for w in items:
                nw = normalize_text(w)
                if nw and nw not in combined:
                    combined.append(nw)
        if context_text:
            for auto_w in extract_priority_keywords(context_text, max_keywords=42):
                if auto_w not in combined:
                    combined.append(auto_w)
        return combined[:65]

    def _learned_prior_bonus(self, misheard: str, target: str) -> float:
        return 0.12 if f"{misheard}->{target}" in self.learned_confusions.get("word_pairs", {}) else 0.0

    def transcribe_baseline(self, audio_16k) -> str:
        chunks = preprocess_and_chunk_audio(np.asarray(audio_16k, dtype=np.float32))
        if not chunks:
            return ""
        parts = []
        for ch in chunks:
            feats = self.processor(ch, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            with torch.no_grad():
                ids = self.baseline_model.generate(feats, max_new_tokens=128)
            txt = normalize_text(self.processor.batch_decode(ids, skip_special_tokens=True)[0])
            if txt and not (len(chunks) == 1 and len(ch) < 24000 and txt in HALLUCINATION_PHRASES):
                parts.append(txt)
        return " ".join(parts).strip()

    def _extract_word_confidences(self, out_obj, chosen_beam_idx: int) -> dict:
        conf_map = {}
        try:
            trans_scores = self.ft_model.compute_transition_scores(
                out_obj.sequences, out_obj.scores, out_obj.beam_indices, normalize_logits=True
            )
            gen_len = len(out_obj.scores)
            gen_ids = out_obj.sequences[chosen_beam_idx, -gen_len:].cpu().tolist()
            probs = torch.exp(trans_scores[chosen_beam_idx]).clamp(0.0, 1.0).cpu().tolist()
            special_ids = set(self.processor.tokenizer.all_special_ids)

            buckets = []
            for tid, pr in zip(gen_ids, probs):
                if tid in special_ids:
                    continue
                piece = self.processor.tokenizer.decode([tid])
                if not piece:
                    continue
                if piece.startswith(" ") or not buckets:
                    buckets.append([piece, [pr]])
                else:
                    buckets[-1][0] += piece
                    buckets[-1][1].append(pr)

            for raw_w, pr_list in buckets:
                nw = normalize_text(raw_w)
                if nw:
                    conf_map[nw] = min(pr_list)
        except Exception:
            pass
        return conf_map

    def _rescue_acronyms(self, words: list, acronyms: list) -> tuple:
        if not acronyms or len(words) < 2:
            return words, []
        rescued, out_words = [], []
        ac_set = {a.lower() for a in acronyms if 2 <= len(a) <= 5}
        i = 0
        while i < len(words):
            matched_len, matched_ac = 0, None
            for span in (5, 4, 3, 2):
                if i + span <= len(words):
                    seg = words[i : i + span]
                    if all(len(tok) == 1 for tok in seg) and "".join(seg) in ac_set:
                        matched_len, matched_ac = span, "".join(seg)
                        break
            if matched_ac:
                rescued.append((" ".join(words[i : i + matched_len]), matched_ac))
                out_words.append(matched_ac)
                i += matched_len
            else:
                out_words.append(words[i])
                i += 1
        return out_words, rescued

    def _apply_priority_hotwords(
        self, sentence: str, hotwords: list, acronyms: list, context_bigrams: set,
        word_confs: dict, slide_topic: str = "", use_confidence_gate: bool = True, gate_thresh: float = 0.88,
    ):
        if not hotwords or not sentence:
            return sentence, []

        words, rescued = self._rescue_acronyms(sentence.split(), acronyms)
        single_hotwords = [h for h in hotwords if " " not in h]
        phrase_hotwords = [tuple(h.split()) for h in hotwords if len(h.split()) == 2]
        target_pairs = list(dict.fromkeys(phrase_hotwords + list(context_bigrams)))

        merged_words, idx = [], 0
        while idx < len(words):
            if idx + 1 < len(words):
                w1, w2 = words[idx], words[idx + 1]
                pair_conf = min(word_confs.get(w1, 0.65), word_confs.get(w2, 0.65))
                if not (use_confidence_gate and pair_conf >= gate_thresh + 0.02):
                    joined = w1 + w2
                    matched_comp = next(
                        (hw for hw in single_hotwords if len(hw) >= 6 and abs(len(hw) - len(joined)) <= 2 and phonetic_similarity(joined, hw) >= 0.85),
                        None,
                    )
                    if matched_comp and matched_comp != joined:
                        rescued.append((f"{w1} {w2}", matched_comp))
                        merged_words.append(matched_comp)
                        word_confs[matched_comp] = 0.95
                        idx += 2
                        continue
            merged_words.append(words[idx])
            idx += 1
        words = merged_words

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if (w1, w2) in target_pairs:
                continue
            pair_conf = min(word_confs.get(w1, 0.65), word_confs.get(w2, 0.65))
            if use_confidence_gate and pair_conf >= gate_thresh:
                continue
            for p1, p2 in target_pairs:
                if p1 in STOPWORDS and p2 in STOPWORDS:
                    continue
                if abs(len(p1) - len(w1)) <= 2 and abs(len(p2) - len(w2)) <= 2:
                    sim1, sim2 = phonetic_similarity(w1, p1), phonetic_similarity(w2, p2)
                    if min(sim1, sim2) >= 0.78 and ((sim1 + sim2) / 2.0) >= 0.85:
                        if f"{w1} {w2}" != f"{p1} {p2}":
                            rescued.append((f"{w1} {w2}", f"{p1} {p2}"))
                            words[i], words[i + 1] = p1, p2
                        break

        pending_swaps = []
        for i, orig_w in enumerate(words):
            if orig_w in STOPWORDS or orig_w in single_hotwords:
                continue
            w_conf = word_confs.get(orig_w, 0.65)
            if use_confidence_gate and w_conf >= gate_thresh:
                continue

            best_hw, best_sim, has_bg = None, 0.0, False
            for hw in single_hotwords:
                if abs(len(hw) - len(orig_w)) > 3:
                    continue
                sim = phonetic_similarity(orig_w, hw) + self._learned_prior_bonus(orig_w, hw)
                prev_ok = (i > 0 and (words[i - 1], hw) in context_bigrams)
                next_ok = (i < len(words) - 1 and (hw, words[i + 1]) in context_bigrams)
                eff_sim = sim + (0.08 if (prev_ok or next_ok) else 0.0)
                if eff_sim > best_sim:
                    best_sim, best_hw, has_bg = eff_sim, hw, (prev_ok or next_ok)

            sim_threshold = 0.77 if w_conf < 0.55 else 0.82
            if best_hw and best_sim >= sim_threshold:
                pending_swaps.append((i, orig_w, best_hw, best_sim, has_bg))

        if pending_swaps:
            base_sent = " ".join(words)
            trial_sents = []
            for i, _, best_hw, _, _ in pending_swaps:
                tw = words.copy()
                tw[i] = best_hw
                trial_sents.append(" ".join(tw))

            all_scores = self.score_sentences_batched([base_sent] + trial_sents, slide_topic=slide_topic)
            base_fluency = all_scores[0]
            for (i, orig_w, best_hw, best_sim, has_bg), new_fluency in zip(pending_swaps, all_scores[1:]):
                if has_bg or best_sim >= 0.91 or (new_fluency >= base_fluency - 0.45):
                    rescued.append((orig_w, best_hw))
                    words[i] = best_hw

        return " ".join(words), rescued

    def _transcribe_single_chunk(
        self, chunk_16k, hotwords, acronyms, context_bigrams, slide_topic, alpha, hw_weight, gate_thresh, use_confidence_gate
    ):
        feats = self.processor(chunk_16k, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        gen_kwargs = {
            "num_beams": 5, "num_return_sequences": 5, "max_new_tokens": 128,
            "return_dict_in_generate": True, "output_scores": True,
        }
        if hotwords:
            try:
                p_ids = self.processor.get_prompt_ids("Presentation vocabulary: " + ", ".join(hotwords[:25]), return_tensors="pt").to(self.device)
                gen_kwargs["prompt_ids"] = (p_ids.squeeze(0) if p_ids.ndim > 1 else p_ids)[-120:]
            except Exception:
                pass

        with torch.no_grad():
            try:
                out = self.ft_model.generate(feats, **gen_kwargs)
            except Exception:
                gen_kwargs.pop("prompt_ids", None)
                out = self.ft_model.generate(feats, **gen_kwargs)

        raw_cands = [normalize_text(c) for c in self.processor.batch_decode(out.sequences, skip_special_tokens=True)]
        ac_scores = out.sequences_scores.cpu().tolist()
        lm_scores = self.score_sentences_batched(raw_cands, slide_topic=slide_topic)

        best_idx, best_sent, best_total = 0, raw_cands[0], -1e9
        hw_set = set(hotwords)
        for b_idx, (norm_c, ac_sc, lm_sc) in enumerate(zip(raw_cands, ac_scores, lm_scores)):
            c_words = norm_c.split()
            hw_hits = sum(1 for w in c_words if w in hw_set)
            bg_hits = sum(1 for k in range(len(c_words) - 1) if (c_words[k], c_words[k + 1]) in context_bigrams)
            total = ac_sc + (alpha * lm_sc) + (hw_weight * hw_hits) + (0.25 * bg_hits)
            if total > best_total:
                best_total, best_sent, best_idx = total, norm_c, b_idx

        if len(chunk_16k) < 24000 and best_sent in HALLUCINATION_PHRASES:
            return "", [], {}

        word_confs = self._extract_word_confidences(out, best_idx)
        final_sent, rescued_pairs = self._apply_priority_hotwords(
            best_sent, hotwords, acronyms, context_bigrams, word_confs,
            slide_topic=slide_topic, use_confidence_gate=use_confidence_gate, gate_thresh=gate_thresh,
        )
        return final_sent, rescued_pairs, word_confs

    def transcribe_accentbridge(
        self, audio_16k, hotwords_raw=None, context_text: str = "", slide_topic: str = "",
        alpha: float = None, hw_weight: float = None, gate_thresh: float = None, use_confidence_gate: bool = True,
    ):
        alpha = self.calib["alpha"] if alpha is None else alpha
        hw_weight = self.calib["hw_weight"] if hw_weight is None else hw_weight
        gate_thresh = self.calib["gate_threshold"] if gate_thresh is None else gate_thresh

        chunks = preprocess_and_chunk_audio(np.asarray(audio_16k, dtype=np.float32))
        if not chunks:
            self.last_word_confidences = {}
            return "", []

        hotwords = self.parse_hotwords(hotwords_raw, context_text=context_text)
        acronyms = extract_acronyms(context_text)
        context_bigrams = extract_context_bigrams(context_text)
        if not slide_topic and context_text:
            slide_topic = context_text.splitlines()[0][:60]

        all_sents, all_rescued, merged_confs = [], [], {}
        for ch in chunks:
            s_txt, resc, w_confs = self._transcribe_single_chunk(
                ch, hotwords, acronyms, context_bigrams, slide_topic, alpha, hw_weight, gate_thresh, use_confidence_gate
            )
            if s_txt:
                all_sents.append(s_txt)
                all_rescued.extend(resc)
                merged_confs.update(w_confs)

        self.last_word_confidences = merged_confs
        return " ".join(all_sents).strip(), all_rescued
