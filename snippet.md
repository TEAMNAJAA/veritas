# AccentBridge: Phoneme-Aligned & Slide-Synced Subtitle AI
**Track:** AI + Languages — *Veritas AI International High School Hackathon 2026*

## Overview
Non-native English presenters (especially Southeast and East Asian speakers) routinely suffer from subtitle errors on domain terms due to systematic L2 phonological shifts (*th-stopping*, *r/l* conflation, and final consonant-cluster deletion). **AccentBridge** combines acoustic Whisper adaptation with per-slide phoneme-level listening.

## Key Technical Contributions
1. **Selective Attention Adaptation (11 MB Delta):** Adapts upper encoder and decoder attention projections of `openai/whisper-tiny.en` with SpecAugment on real Zoom conversations from **EdAcc**.
2. **CMU ARPAbet Articulatory Phoneme Alignment:** Aligns spoken words and slide vocabulary in phoneme space using the 134,000-word CMU Pronouncing Dictionary plus data-mined L2 confusion priors.
3. **Dev-Calibrated Acoustic Confidence Gate:** Calibrates rescoring weights on a held-out dev split and locks high-certainty spoken words to prevent wrong-slide hallucinations.
4. **Slide-Synced Presentation UI:** Parses `.pptx` (with Speaker Notes), `.pdf`, and `.docx` per slide with Manual Prev/Next/Q&A Jump, Voice Auto-Advance, Visual Word Diff, Long-Audio Pause Chunking, and `.SRT` export.

## Benchmark Results (Unseen Test Speakers)
![Main Benchmark](results/benchmark_chart.png)
![Per-Accent Breakdown](results/per_accent_wer.png)
*(Full paired bootstrap 95% confidence intervals and p-values are generated in `results/report.md`)*

## Quickstart
```bash
pip install -r requirements.txt
python verify_project.py     # Sanity check all modules in 5 seconds
python make_demo_slides.py   # Generate sample 3-slide PowerPoint deck
python train_and_eval.py     # Fine-tune, calibrate on dev, and save charts + report
python app.py                # Launch live Gradio web app
