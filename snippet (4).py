from pptx import Presentation

def create_sample_pptx(filename: str = "demo_slides.pptx"):
    prs = Presentation()
    slides_content = [
        (
            "AccentBridge: API & File Architecture",
            "• Check the audio file on the right slide\n• Three core API modules in our open source pipeline\n• Built for live English presentations",
            "Today please look at the file on the right slide where we present three core API modules."
        ),
        (
            "Dataset & WER Evaluation",
            "• Evaluated on unseen speakers from EdAcc\n• Measures WER and target keyword recall on GPU\n• Compares zero-shot Whisper against AccentBridge",
            "Next slide. Here we compare our baseline WER and target keyword recall across Asian accents."
        ),
        (
            "Failure Cases & Future Strategy",
            "• Over-biasing on short out-of-context homophones\n• Rapid overlapping speech in group calls\n• Next step: low-latency WebSocket streaming",
            "Finally we analyze where our algorithm fails on overlapping speech and explain our future strategy."
        ),
    ]

    for title_txt, body_txt, notes_txt in slides_content:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title_txt
        slide.placeholders[1].text = body_txt
        slide.notes_slide.notes_text_frame.text = notes_txt

    prs.save(filename)
    print(f"✅ สร้างไฟล์ {filename} สำเร็จ!")

if __name__ == "__main__":
    create_sample_pptx()
