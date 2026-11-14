import os
import re
from collections import Counter

FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "into", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "once", "here", "there",
    "all", "any", "both", "each", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "why", "where", "also", "using", "used"
}

COMMON_CAPS_IGNORE = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "SLIDE", "NEXT", "TODAY", "OVERVIEW", "SUMMARY"
}

VOICE_ADVANCE_CUES = [
    "next slide", "moving on to", "move to the next", "on the following slide", "turning to the next"
]

def accent_difficulty_bonus(word: str) -> float:
    score = 0.0
    if any(pat in word for pat in ["th", "ght", "str", "spr", "spl", "shr", "v", "z"]):
        score += 2.0
    if "r" in word or "l" in word:
        score += 1.2
    if re.search(r"(ld|st|nd|nt|ct|pt|ft|le|ce|se)$", word):
        score += 1.5
    if len(word) >= 7:
        score += 1.0
    return score

def extract_acronyms(raw_text: str) -> list:
    if not raw_text:
        return []
    found = re.findall(r"\b[A-Z]{2,6}\b", raw_text)
    acronyms = []
    for ac in found:
        if ac not in COMMON_CAPS_IGNORE and ac.lower() not in acronyms:
            acronyms.append(ac.lower())
    return acronyms[:15]

def extract_priority_keywords(raw_text: str, max_keywords: int = 35) -> list:
    if not raw_text:
        return []
    acronyms = extract_acronyms(raw_text)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", raw_text.lower())
    filtered = [t.strip("'-") for t in tokens if t.strip("'-") not in FUNCTION_WORDS and len(t.strip("'-")) >= 3]

    counts = Counter(filtered)
    scored = [((freq * 1.3) + accent_difficulty_bonus(word), word) for word, freq in counts.items()]
    scored.sort(key=lambda x: x[0], reverse=True)

    merged = list(dict.fromkeys(acronyms + [w for _, w in scored]))
    return merged[:max_keywords]

def extract_context_bigrams(raw_text: str) -> set:
    if not raw_text:
        return set()
    clean_words = re.findall(r"[a-z0-9']+", raw_text.lower())
    if len(clean_words) < 2:
        return set()
    return {(clean_words[i], clean_words[i + 1]) for i in range(len(clean_words) - 1)}

def _build_slide_records(raw_pages: list) -> list:
    slides = []
    for idx, txt in enumerate(raw_pages):
        cleaned = (txt or "").strip()
        if not cleaned:
            cleaned = f"(Slide {idx + 1} - No text content)"
        first_line = cleaned.splitlines()[0].strip()[:65]
        slides.append({
            "index": idx,
            "title": first_line or f"Slide {idx + 1}",
            "text": cleaned,
            "keywords": extract_priority_keywords(cleaned, max_keywords=32),
            "acronyms": extract_acronyms(cleaned),
        })
    return slides

def extract_slides_from_file(file_obj) -> list:
    if file_obj is None:
        return []
    filepath = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", "")
    if not filepath or not os.path.exists(filepath):
        return []

    ext = os.path.splitext(filepath)[1].lower()
    pages = []

    try:
        if ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(filepath)
            for slide in prs.slides:
                parts = []
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        parts.append(shape.text_frame.text)
                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            for cell in row.cells:
                                parts.append(cell.text)
                if getattr(slide, "has_notes_slide", False) and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text
                    if notes and notes.strip():
                        parts.append("[Speaker Notes] " + notes)
                pages.append("\n".join(p for p in parts if p and p.strip()))

        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            for page in reader.pages:
                pages.append(page.extract_text() or "")

        elif ext == ".docx":
            import docx
            doc = docx.Document(filepath)
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for i in range(0, len(paras), 4):
                pages.append("\n".join(paras[i : i + 4]))

        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            blocks = [b.strip() for b in re.split(r"\n\s*---\s*\n|\n\s*\n", raw) if b.strip()]
            pages.extend(blocks)
    except Exception as e:
        return [{"index": 0, "title": "Error", "text": f"Failed to read file: {e}", "keywords": [], "acronyms": []}]

    return _build_slide_records(pages)

def extract_text_from_file(file_obj) -> str:
    slides = extract_slides_from_file(file_obj)
    return "\n\n".join(s["text"] for s in slides)

def get_distinctive_next_keywords(curr_slide: dict, next_slide: dict, limit: int = 8) -> list:
    if not next_slide:
        return []
    curr_set = set(curr_slide.get("keywords", [])) if curr_slide else set()
    return [w for w in next_slide.get("keywords", []) if w not in curr_set][:limit]

def should_auto_advance(spoken_text: str, curr_slide: dict, next_slide: dict, sim_fn) -> tuple:
    if not spoken_text or not next_slide:
        return False, ""

    norm_spoken = spoken_text.lower().strip()
    for cue in VOICE_ADVANCE_CUES:
        if cue in norm_spoken:
            return True, f"🔄 เลื่อนหน้าอัตโนมัติ (ตรวจพบคำสั่งเสียง '{cue}')"

    spoken_words = [w for w in norm_spoken.split() if len(w) >= 3 and w not in FUNCTION_WORDS]
    if len(spoken_words) < 3:
        return False, ""

    curr_kws = set(curr_slide.get("keywords", []))
    next_distinct_kws = [w for w in next_slide.get("keywords", []) if w not in curr_kws and len(w) >= 3]
    curr_distinct_kws = [w for w in curr_kws if w not in set(next_slide.get("keywords", [])) and len(w) >= 3]

    matched_next = {nkw for sw in spoken_words for nkw in next_distinct_kws if sw == nkw or sim_fn(sw, nkw) >= 0.86}
    matched_curr = {ckw for sw in spoken_words for ckw in curr_distinct_kws if sw == ckw or sim_fn(sw, ckw) >= 0.86}

    if len(matched_next) >= 2 and len(matched_next) > len(matched_curr):
        matched_str = ", ".join(sorted(matched_next)[:4])
        return True, f"🔄 เลื่อนไปหน้า {next_slide['index'] + 1} อัตโนมัติ (พบคำเฉพาะหน้าถัดไป: {matched_str})"

    return False, ""
