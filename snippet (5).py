import os
import numpy as np
from accent_engine import phonetic_similarity, normalize_audio_safe
from slide_parser import extract_priority_keywords, extract_acronyms

def run_checks():
    print("🔍 กำลังตรวจสอบความสมบูรณ์ของโปรเจกต์ทั้ง 9 ไฟล์...")
    required = [
        ".gitignore", "requirements.txt", "slide_parser.py", "accent_engine.py",
        "train_and_eval.py", "app.py", "make_demo_slides.py", "verify_project.py", "README.md"
    ]
    missing = [f for f in required if not os.path.exists(f)]
    assert not missing, f"❌ ขาดไฟล์: {missing}"

    sim_homophone = phonetic_similarity("write", "right")
    sim_l2_shift = phonetic_similarity("slight", "slide")
    sim_unrelated = phonetic_similarity("fine", "file")
    assert sim_homophone >= 0.95, "❌ พจนานุกรมเสียง CMU พ้องเสียงผิดพลาด"
    assert sim_l2_shift >= 0.82 and sim_unrelated < 0.78, "❌ เกณฑ์แยกแยะเสียงสำเนียงเอเชียผิดพลาด"
    print(f"✅ พจนานุกรมเสียงผ่าน! write↔right={sim_homophone:.2f} | slight↔slide={sim_l2_shift:.2f} | fine↔file={sim_unrelated:.2f}")

    kws = extract_priority_keywords("Check API and WER on the right slide")
    acs = extract_acronyms("Check API and WER on the right slide")
    assert "api" in acs and "wer" in acs and "slide" in kws, "❌ ตัวสกัดคำศัพท์และตัวย่อผิดพลาด"
    print("✅ ตัวสกัดสไลด์และตัวย่อผ่าน!")

    dummy_audio = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)
    dummy_audio[5] = 40.0
    assert np.max(np.abs(normalize_audio_safe(dummy_audio, 16000))) <= 1.0
    print("✅ ระบบป้องกันเสียงกระแทกไมค์ผ่าน!")
    print("🎉 ทุกไฟล์พร้อม Commit ขึ้น GitHub 100% ค่ะ!")

if __name__ == "__main__":
    run_checks()
