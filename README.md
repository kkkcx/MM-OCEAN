<div align="center">

# MM-OCEAN

**A Multi-Granularity Benchmark for Grounded Personality Reasoning**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Dataset on HuggingFace](https://img.shields.io/badge/🤗%20Dataset-MM--OCEAN-yellow)](https://huggingface.co/datasets/anonymous-mm-ocean/MM-OCEAN)
[![Code on GitHub](https://img.shields.io/badge/GitHub-MM--OCEAN-blue?logo=github)](https://github.com/kkkcx/MM-OCEAN)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-D%26B%20Track-green)](https://neurips.cc/Conferences/2026)

</div>

## Overview

MM-OCEAN is the first benchmark for **Grounded Personality Reasoning (GPR)**, evaluating whether Multimodal LLMs can ground Big Five personality judgments in observable behavioral evidence. Across 27 evaluated MLLMs we find that **51% of correct ratings are not grounded in retrieved cues** — a phenomenon we term the *Prejudice Gap*.

| Component | Count |
|---|---|
| Test videos (15-second clips from ChaLearn FI V2) | **1,104** |
| Human-verified atomic behavioral observations | ~13,500 |
| Evidence-grounded trait analyses (5 traits × videos) | 5,520 |
| Cue-grounding MCQs (7 cognitive categories) | **5,320** |

## Three-Tier Evaluation

| Task | Description |
|---|---|
| **T1 — Personality Rating** | Predict Big Five trait levels on a 5-point ordinal scale |
| **T2 — Open-Ended Reasoning** | Generate evidence-grounded rationales; scored by an AI-as-Judge on 4 dimensions |
| **T3 — Grounding MCQs** | Answer cue-grounding multiple-choice questions across 7 cognitive categories |

## Four Failure-Mode Metrics

| Metric | Meaning |
|---|---|
| **PR** (Prejudice Rate) | Correct rating without grounded cues — the headline failure mode |
| **CR** (Confabulation Rate) | Correct rating with incoherent reasoning |
| **IR** (Integration-failure Rate) | Correct cues but wrong rating |
| **HR** (Holistic-Grounding Rate) | All three tasks correct on the same sample — our principal metric |

## Repository Structure

```
MM-OCEAN/
├── data/test/          # 1,104 annotation JSONs (one per video)
├── prompts/
│   ├── unified.py      # Three-task evaluation prompt template
│   ├── judge.py        # AI-as-Judge prompt and 4-dimension rubric
│   ├── observer.py     # Observer agent prompt (Stage 1)
│   ├── psychologist.py # Psychologist agent prompt (Stage 2)
│   ├── examiner.py     # Examiner agent prompt (Stage 3)
│   └── aligner.py      # Aligner agent prompt (Stage 4)
├── evaluate.py         # Self-contained scoring script
├── README.md
├── LICENSE
└── croissant.json      # NeurIPS-compliant dataset metadata
```

Each annotation JSON contains:

```json
{
  "video_id": "abc.mp4",
  "transcription": "...",
  "original_scores": {"extraversion": 0.62, "...": "..."},
  "observations": [
    {"dimension": "Expression", "start_time": 2.7, "end_time": 4.9,
     "description": "...", "bboxes": [[0.3, 0.4, 0.5, 0.6]]}
  ],
  "personality_analyses": [{"trait": "extraversion", "level": "High", "rationale": "..."}],
  "mcq_questions": [
    {"category": "Personality Attribution",
     "question": "...", "options": ["..."],
     "correct_answer": "C", "explanation": "..."}
  ]
}
```

## Video Access

MM-OCEAN annotations are built on top of **ChaLearn First Impressions V2** videos. Due to licensing, we do not redistribute the videos.

1. Visit the [ChaLearn First Impressions V2](https://chalearnlap.cvc.uab.es/dataset/24/description/) page.
2. Request access and download the test split.
3. Place the `.mp4` files so that `video_id` fields in our JSONs match the filenames.

## Quick Start

### Step 1. Run your model

Use `prompts/unified.py` as the per-video prompt. Save outputs as JSONL:

```json
{"video_id": "abc.mp4",
 "task1": {"extraversion": "High", "...": "..."},
 "task2": {"extraversion": "The person ...", "...": "..."},
 "task3": [{"q_idx": 1, "answer": "C"}]}
```

### Step 2. Run the AI-as-Judge for Task 2

Apply `prompts/judge.py` to score each Task 2 explanation and add a `task2_judged` field to your JSONL.

### Step 3. Evaluate

```bash
python evaluate.py --gt_dir data/test --pred_file results/my_model.jsonl
```

Outputs include T1 accuracy / MAE, T2 composite, T3 overall and per-category accuracy, and the four failure-mode rates.

## Citation

```bibtex
@inproceedings{mmocean2026,
  title     = {Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026}
}
```

## License

Released under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The underlying videos follow the ChaLearn First Impressions V2 license. Intended for academic research only; see the paper appendix (*Ethics and Responsible Use*) for detailed guidelines.
