import os
import re
import json
from collections import Counter
from difflib import SequenceMatcher
import numpy as np
import soundfile as sf
import gradio as gr
from accent_engine import AccentBridgeEngine, phonetic_similarity, normalize_audio_safe, STOPWORDS
from slide_parser import (
    extract_slides_from_file,
    extract_priority_keywords,
    extract_acronyms,
    get_distinctive_next_keywords,
    should_auto_advance,
)

engine = AccentBridgeEngine(finetuned_dir="./saved_model")

SAMPLE_DECK_TEXT = """Slide 1: Project Overview & API Architecture
Today we present our baseline model on the right slide and explain three core API modules in our open source file system.
---
Slide 2: Dataset & WER Evaluation
Next we evaluate WER and keyword recall across Asian accents using unseen test speakers on GPU.
---
Slide 3: Failure Cases & Future Strategy
Finally we analyze where the algorithm fails on rapid overlapping speech and outline our future strategy."""

def load_judge_demo_clips() -> list:
    manifest_path = "./demo_samples/manifest.json"
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        return [x for x in items if os.path.exists(x.get("file", ""))]
    except Exception:
        return []

def format_srt_time(seconds: float) -> str:
    hrs, mins, secs = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def export_srt_file(history: list) -> str:
    path = "./accentbridge_subtitles.srt"
    lines, cursor = [], 0.0
    for idx, item in enumerate(history, start=1):
        dur = max(2.0, item.get("duration", 4.0))
        lines.append(f"{idx}\n{format_srt_time(cursor)} --> {format_srt_time(cursor + dur)}\n{item['text']}\n")
        cursor += dur + 0.3
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) if lines else "1\n00:00:00,000 --> 00:00:02,000\nNo subtitles recorded yet.\n")
    return path

def build_dropdown_choices(slides: list) -> list:
    return [f"หน้าที่ {i + 1}: {s['title'][:45]}" for i, s in enumerate(slides)]

def render_slide_card(slides: list, idx: int) -> str:
    if not slides:
        return "📌 **ยังไม่ได้โหลดสไลด์** — กดปุ่ม **'🎯 โหลดสไลด์ตัวอย่าง 3 หน้า'** ด้านบนเพื่อทดลองทันที หรืออัปโหลดไฟล์ของคุณเองได้เลยค่ะ"
    idx = max(0, min(idx, len(slides) - 1))
    s = slides[idx]
    ac_note = f" | 🔤 ตัวย่อบนสไลด์: {', '.join(a.upper() for a in s.get('acronyms', []))}" if s.get("acronyms") else ""
    return f"### 📑 สไลด์หน้าที่ {idx + 1} / {len(slides)}: {s['title']}{ac_note}\n\n{s['text'][:500]}"

def render_word_diff_html(base_txt: str, ab_txt: str) -> str:
    if not base_txt and not ab_txt:
        return "<div style='color:#64748b;padding:10px;background:#f8fafc;border-radius:8px;'><b>Visual Word Diff:</b> รอเปรียบเทียบคำผิด (ขีดฆ่าสีแดง) กับคำที่แก้สำเร็จ (สีเขียว)...</div>"
    b_words, a_words = base_txt.split(), ab_txt.split()
    chunks = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, b_words, a_words).get_opcodes():
        if tag == "equal":
            chunks.append(" ".join(a_words[j1:j2]))
        else:
            if i1 < i2:
                chunks.append(f"<del style='background:#fee2e2;color:#991b1b;padding:1px 6px;border-radius:4px;'>{' '.join(b_words[i1:i2])}</del>")
            if j1 < j2:
                chunks.append(f"<span style='background:#dcfce7;color:#166534;padding:1px 6px;border-radius:4px;font-weight:bold;'>{' '.join(a_words[j1:j2])} ✨</span>")
    return "<div style='font-size:1.08rem;line-height:1.8;padding:10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'><b>Visual Diff (Baseline → AccentBridge):</b><br>" + " ".join(chunks) + "</div>"

def load_presentation(slide_file, rough_script):
    slides = extract_slides_from_file(slide_file)
    if not slides and rough_script and rough_script.strip():
        blocks = [b.strip() for b in re.split(r"\n\s*---\s*\n|\n\s*\n", rough_script) if b.strip()]
        slides = [
            {"index": i, "title": b.splitlines()[0][:65], "text": b, "keywords": extract_priority_keywords(b, 32), "acronyms": extract_acronyms(b)}
            for i, b in enumerate(blocks)
        ]
    if not slides:
        return [], 0, render_slide_card([], 0), "file, right, slide, three, write, open source, api, wer", gr.update(choices=[], value=None)
    choices = build_dropdown_choices(slides)
    return slides, 0, render_slide_card(slides, 0), ", ".join(slides[0]["keywords"]), gr.update(choices=choices, value=choices[0])

def load_sample_deck():
    slides, idx, card, kws, dd_update = load_presentation(None, SAMPLE_DECK_TEXT)
    return SAMPLE_DECK_TEXT, slides, idx, card, kws, dd_update

def pick_judge_clip(clip_idx: int):
    clips = load_judge_demo_clips()
    if clip_idx < len(clips):
        c = clips[clip_idx]
        wav_arr, sr = sf.read(c["file"], dtype="float32")
        int16_arr = (np.clip(wav_arr, -1.0, 1.0) * 32767).astype(np.int16)
        return (sr, int16_arr), c["keywords"]
    return None, "file, right, slide, three, model"

def save_current_keywords(slides: list, idx: int, edited_keywords: str) -> list:
    if slides and 0 <= idx < len(slides):
        slides[idx]["keywords"] = engine.parse_hotwords(edited_keywords)
    return slides

def change_slide_manual(slides: list, current_idx: int, target_idx: int, current_keywords: str):
    if not slides:
        return slides, 0, render_slide_card([], 0), current_keywords, gr.update()
    slides = save_current_keywords(slides, current_idx, current_keywords)
    new_idx = max(0, min(target_idx, len(slides) - 1))
    choices = build_dropdown_choices(slides)
    return slides, new_idx, render_slide_card(slides, new_idx), ", ".join(slides[new_idx]["keywords"]), gr.update(choices=choices, value=choices[new_idx])

def jump_to_slide_dd(selected_label: str, slides: list, current_idx: int, current_keywords: str):
    if not slides or not selected_label:
        return slides, current_idx, render_slide_card(slides, current_idx), current_keywords
    choices = build_dropdown_choices(slides)
    target_idx = choices.index(selected_label) if selected_label in choices else current_idx
    slides = save_current_keywords(slides, current_idx, current_keywords)
    return slides, target_idx, render_slide_card(slides, target_idx), ", ".join(slides[target_idx]["keywords"])

def extra_sensitivity_adjust(text: str, rescued: list, hotwords: list, mode: str, word_confs: dict):
    if mode == "Conservative (แก้เฉพาะคำที่เสียงใกล้เคียงมาก)":
        filtered_rescued = [(o, n) for o, n in rescued if phonetic_similarity(o, n) >= 0.87]
        reverted = {n: o for o, n in rescued if (o, n) not in filtered_rescued}
        return " ".join([reverted.get(w, w) for w in text.split()]), filtered_rescued
    if mode == "Aggressive (ช่วยสำเนียงไม่ชัดเต็มที่)":
        words, single_hw, new_rescued = text.split(), [h for h in hotwords if " " not in h], list(rescued)
        already_fixed = {w_part for _, n in rescued for w_part in n.split()}
        for i, w in enumerate(words):
            if w in STOPWORDS or w in single_hw or w in already_fixed or word_confs.get(w, 0.65) >= 0.92:
                continue
            best_hw, best_sim = None, 0.0
            for hw in single_hw:
                if abs(len(hw) - len(w)) <= 3:
                    sim = phonetic_similarity(w, hw)
                    if sim > best_sim:
                        best_sim, best_hw = sim, hw
            if best_hw and best_sim >= 0.74:
                words[i] = best_hw
                new_rescued.append((w, best_hw))
        return " ".join(words), new_rescued
    return text, rescued

def render_live_banner(text: str, rescued_pairs: list, hotwords: list, word_confs: dict) -> str:
    rescued_targets = {part for _, hw in rescued_pairs for part in hw.split()}
    hw_set = {part for hw in hotwords for part in hw.split()}
    styled_words = []
    for w in text.split():
        conf = word_confs.get(w, 0.75)
        if w in rescued_targets:
            styled_words.append(f"<span style='background:#22c55e;color:#ffffff;padding:2px 8px;border-radius:6px;'>{w} ✨</span>")
        elif w in hw_set:
            styled_words.append(f"<span style='background:#3b82f6;color:#ffffff;padding:2px 8px;border-radius:6px;'>{w}</span>")
        elif conf < 0.50 and w not in STOPWORDS:
            styled_words.append(f"<span style='border-bottom:3px dashed #f59e0b;color:#fde68a;'>{w}</span>")
        else:
            styled_words.append(w)
    return (
        "<div style='background:#0f172a;color:#f8fafc;padding:18px 24px;border-radius:12px;"
        "font-size:1.45rem;font-weight:600;text-align:center;line-height:1.75;box-shadow:0 4px 12px rgba(0,0,0,0.18);'>"
        + (" ".join(styled_words) if styled_words else "🎙️ รอรับเสียงพูดของคุณ... (เขียว ✨ = แก้คำเพี้ยนด้วย CMU Phonemes | ฟ้า = คำตรงสไลด์)")
        + "</div>"
    )

def format_coaching_summary(history: list) -> str:
    all_pairs = [pair for h in history for pair in h.get("rescued", [])]
    if not all_pairs:
        return "ยังไม่มีคำที่ต้องช่วยแก้"
    counts = Counter(all_pairs)
    return "💡 สรุปคำศัพท์เฉพาะตัวที่ผู้พูดควรเน้นเสียง:\n" + "\n".join(
        [f"• คำว่า '{target}' (ได้ยินคล้าย '{heard}' — แก้ช่วย {cnt} ครั้ง)" for (heard, target), cnt in counts.most_common(6)]
    )

def process_speech(audio_tuple, custom_keywords, rough_script, slides, current_idx, auto_advance_on, sensitivity_mode, gate_on, history):
    if audio_tuple is None:
        return (
            render_live_banner("", [], [], {}), render_word_diff_html("", ""), "", slides, current_idx,
            render_slide_card(slides, current_idx), custom_keywords, gr.update(),
            history, "ยังไม่มีประวัติการพูด", format_coaching_summary(history), None,
        )

    slides = save_current_keywords(slides, current_idx, custom_keywords)
    sr, raw_arr = audio_tuple
    arr = normalize_audio_safe(raw_arr, sr)
    duration_sec = len(arr) / 16000.0

    curr_slide = slides[current_idx] if (slides and 0 <= current_idx < len(slides)) else None
    next_slide = slides[current_idx + 1] if (slides and auto_advance_on and current_idx + 1 < len(slides)) else None

    active_context = curr_slide["text"] if curr_slide else (rough_script or "")
    slide_topic = curr_slide["title"] if curr_slide else ""
    lookahead_words = get_distinctive_next_keywords(curr_slide, next_slide, limit=8) if next_slide else []
    if next_slide:
        active_context = active_context + "\n" + next_slide["text"]
    combined_hw_str = (custom_keywords + ", " + ", ".join(lookahead_words)).strip(", ") if lookahead_words else custom_keywords

    baseline_txt = engine.transcribe_baseline(arr)
    ab_txt, rescued = engine.transcribe_accentbridge(
        arr, hotwords_raw=combined_hw_str, context_text=active_context, slide_topic=slide_topic, use_confidence_gate=gate_on
    )
    if not ab_txt:
        return (
            render_live_banner("", [], [], {}), render_word_diff_html("", ""), "⚠️ ไม่พบเสียงพูดที่ชัดเจน",
            slides, current_idx, render_slide_card(slides, current_idx), custom_keywords,
            gr.update(), history, "\n".join([f"[Slide {h['slide']}] {h['text']}" for h in history]),
            format_coaching_summary(history), None,
        )

    word_confs = dict(engine.last_word_confidences)
    active_hotwords = engine.parse_hotwords(combined_hw_str, context_text=active_context)
    ab_txt, rescued = extra_sensitivity_adjust(ab_txt, rescued, active_hotwords, sensitivity_mode, word_confs)

    new_idx, updated_keywords, advance_note, dd_update = current_idx, custom_keywords, "", gr.update()
    if curr_slide and next_slide and auto_advance_on:
        should_move, reason = should_auto_advance(ab_txt, curr_slide, next_slide, phonetic_similarity)
        if should_move:
            new_idx = current_idx + 1
            updated_keywords = ", ".join(slides[new_idx]["keywords"])
            choices = build_dropdown_choices(slides)
            dd_update = gr.update(choices=choices, value=choices[new_idx])
            advance_note = reason

    used_slide_num = (new_idx + 1) if advance_note else (current_idx + 1 if curr_slide else 1)
    history = list(history) + [{"slide": used_slide_num, "text": ab_txt, "duration": duration_sec, "rescued": rescued}]
    srt_path = export_srt_file(history)

    banner_html = render_live_banner(ab_txt, rescued, active_hotwords, word_confs)
    diff_html = render_word_diff_html(baseline_txt, ab_txt)
    avg_conf = (sum(word_confs.values()) / len(word_confs) * 100.0) if word_confs else 75.0
    logs = [f"🎙️ ความยาว {duration_sec:.1f}s | ความมั่นใจเฉลี่ย {avg_conf:.0f}%"]
    if rescued:
        logs.append("✅ แก้คำ/ตัวย่อสำเร็จ: " + ", ".join([f"'{o}' → '{n}'" for o, n in rescued]))
    if advance_note:
        logs.append(advance_note)

    return (
        banner_html, diff_html, " | ".join(logs), slides, new_idx,
        render_slide_card(slides, new_idx), updated_keywords, dd_update,
        history, "\n".join([f"[Slide {h['slide']}] {h['text']}" for h in history]),
        format_coaching_summary(history), srt_path,
    )

def clear_transcript_history():
    return [], "ล้างประวัติซับไตเติลแล้ว", "ล้างข้อมูลสรุปคำฝึกออกเสียงแล้ว", None

status_msg = (
    "✅ โหลดโมเดล Fine-tuned AccentBridge + พจนานุกรมเสียง CMU ARPAbet สำเร็จ"
    if engine.using_finetuned
    else "⚡ โหมดพร้อมทดสอบทันที (รัน python train_and_eval.py เพื่อสร้างไฟล์โมเดล Fine-tuned)"
)

with gr.Blocks(title="AccentBridge — Phoneme & Slide-Synced Subtitle AI") as demo:
    slides_state, idx_state, history_state = gr.State([]), gr.State(0), gr.State([])
    gr.Markdown(f"# 🎙️ AccentBridge: Phoneme-Aligned & Slide-Synced Subtitle AI\n**สถานะ:** {status_msg}")
    live_banner = gr.HTML(render_live_banner("", [], [], {}))

    with gr.Accordion("📄 ขั้นตอนที่ 1: โหลดไฟล์ Presentation หรือกดปุ่มโหลดสไลด์ตัวอย่าง 1 คลิก", open=True):
        sample_btn = gr.Button("🎯 โหลดสไลด์ตัวอย่าง 3 หน้าพร้อมทดลองทันที (1-Click Sample Deck)", variant="secondary")
        with gr.Row():
            slide_upload = gr.File(file_types=[".pptx", ".pdf", ".docx", ".txt", ".md"], type="filepath", label="อัปโหลดไฟล์สไลด์ (.pptx, .pdf, .docx, .txt)")
            rough_script_box = gr.Textbox(lines=3, label="หรือวางบทพูดแบ่งหน้าด้วยเครื่องหมาย ---")
        load_btn = gr.Button("📂 แยกหน้าสไลด์จากไฟล์/ข้อความด้านบน")

    with gr.Group():
        gr.Markdown("### 🖥️ ขั้นตอนที่ 2: แผงควบคุมหน้าสไลด์ (กดเปลี่ยนมือ + ข้ามหน้า Q&A + เลื่อนหน้าอัตโนมัติ)")
        slide_display_md = gr.Markdown(render_slide_card([], 0))
        with gr.Row():
            prev_btn = gr.Button("◀ หน้าก่อนหน้า (Prev)")
            slide_jump_dd = gr.Dropdown(choices=[], label="เลือกข้ามไปหน้าสไลด์ทันที (ช่วง Q&A)", interactive=True)
            next_btn = gr.Button("หน้าถัดไป (Next) ▶")
        with gr.Row():
            auto_switch_cb = gr.Checkbox(value=True, label="🔄 เลื่อนหน้าตามเสียงพูดอัตโนมัติ")
            gate_cb = gr.Checkbox(value=True, label="🛡️ เปิด Acoustic Confidence Gate")
        sensitivity_radio = gr.Radio(
            choices=["Conservative (แก้เฉพาะคำที่เสียงใกล้เคียงมาก)", "Balanced (สมดุล แนะนำ)", "Aggressive (ช่วยสำเนียงไม่ชัดเต็มที่)"],
            value="Balanced (สมดุล แนะนำ)", label="🎚️ ระดับความเข้มข้นในการดักคำจากสไลด์",
        )
        keywords_box = gr.Textbox(value="file, right, slide, three, write, open source, api, wer, baseline", label="🎯 คำและตัวย่อเฉพาะของหน้านี้", lines=2)

    sample_btn.click(fn=load_sample_deck, inputs=[], outputs=[rough_script_box, slides_state, idx_state, slide_display_md, keywords_box, slide_jump_dd])
    load_btn.click(fn=load_presentation, inputs=[slide_upload, rough_script_box], outputs=[slides_state, idx_state, slide_display_md, keywords_box, slide_jump_dd])
    prev_btn.click(fn=lambda s, i, k: change_slide_manual(s, i, i - 1, k), inputs=[slides_state, idx_state, keywords_box], outputs=[slides_state, idx_state, slide_display_md, keywords_box, slide_jump_dd])
    next_btn.click(fn=lambda s, i, k: change_slide_manual(s, i, i + 1, k), inputs=[slides_state, idx_state, keywords_box], outputs=[slides_state, idx_state, slide_display_md, keywords_box, slide_jump_dd])
    slide_jump_dd.select(fn=jump_to_slide_dd, inputs=[slide_jump_dd, slides_state, idx_state, keywords_box], outputs=[slides_state, idx_state, slide_display_md, keywords_box])

    gr.Markdown("### 🎤 ขั้นตอนที่ 3: พูดใส่ไมค์ หรือกดปุ่มคลิปเสียงตัวอย่างเพื่อให้กรรมการทดสอบได้ใน 1 คลิก")
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="อัดเสียงพูด หรืออัปโหลดไฟล์เสียง")
        with gr.Column():
            run_btn = gr.Button("🚀 แปลงเสียงเป็นซับไตเติล", variant="primary")
            diff_out = gr.HTML(render_word_diff_html("", ""))
            rescued_log = gr.Textbox(label="🔍 สถานะเสียง & บันทึกการแก้คำ/เปลี่ยนหน้า")

    judge_clips = load_judge_demo_clips()
    if judge_clips:
        with gr.Row():
            clip1_btn = gr.Button(f"🎧 โหลดคลิปเสียงจริงตัวอย่างที่ 1 (สำเนียง {judge_clips[0]['l1']})")
            clip1_btn.click(fn=lambda: pick_judge_clip(0), inputs=[], outputs=[audio_in, keywords_box])
            if len(judge_clips) > 1:
                clip2_btn = gr.Button(f"🎧 โหลดคลิปเสียงจริงตัวอย่างที่ 2 (สำเนียง {judge_clips[1]['l1']})")
                clip2_btn.click(fn=lambda: pick_judge_clip(1), inputs=[], outputs=[audio_in, keywords_box])

    with gr.Accordion("📝 ประวัติซับไตเติลสะสม, คำแนะนำฝึกออกเสียง & ดาวน์โหลดไฟล์ .SRT", open=True):
        history_box = gr.Textbox(lines=4, label="บทบันทึกซับไตเติลเรียงตามหน้าสไลด์", interactive=False)
        coaching_box = gr.Textbox(lines=3, label="🎓 สรุปคำที่ผู้พูดออกเสียงเพี้ยนบ่อย", interactive=False)
        with gr.Row():
            srt_file_out = gr.File(label="📥 ดาวน์โหลดไฟล์ซับไตเติล (.srt)")
            clear_btn = gr.Button("🗑️ ล้างประวัติซับไตเติล")

    speech_inputs = [audio_in, keywords_box, rough_script_box, slides_state, idx_state, auto_switch_cb, sensitivity_radio, gate_cb, history_state]
    speech_outputs = [live_banner, diff_out, rescued_log, slides_state, idx_state, slide_display_md, keywords_box, slide_jump_dd, history_state, history_box, coaching_box, srt_file_out]

    audio_in.stop_recording(fn=process_speech, inputs=speech_inputs, outputs=speech_outputs)
    run_btn.click(fn=process_speech, inputs=speech_inputs, outputs=speech_outputs)
    clear_btn.click(fn=clear_transcript_history, inputs=[], outputs=[history_state, history_box, coaching_box, srt_file_out])

if __name__ == "__main__":
    demo.launch(share=("SPACE_ID" not in os.environ))
