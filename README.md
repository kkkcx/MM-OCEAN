---
license: cc-by-nc-sa-4.0
task_categories:
  - video-classification
  - question-answering
  - visual-question-answering
language:
  - en
tags:
  - personality-perception
  - big-five
  - grounded-reasoning
  - benchmark
  - multimodal
  - video-understanding
pretty_name: "MM-OCEAN: A Benchmark for Grounded Personality Reasoning"
size_categories:
  - 1K<n<10K
---

# MM-OCEAN: A Multi-Granularity Benchmark for Grounded Personality Reasoning

> **Hosting**: dataset on [HuggingFace](https://huggingface.co/datasets/anonymous-mm-ocean/MM-OCEAN). Code on [GitHub](https://github.com/kkkcx/MM-OCEAN). The two repositories carry the same evaluation code, prompts, and benchmark annotations.

## Overview

MM-OCEAN is the first benchmark for **Grounded Personality Reasoning (GPR)**, requiring Multimodal Large Language Models (MLLMs) to ground Big Five personality judgments in observable behavioral evidence. The benchmark evaluates models through three tasks of increasing cognitive depth.

| Component | Count |
|---|---|
| Test videos | 1,104 (15-second clips from ChaLearn First Impressions V2) |
| Atomic behavioral observations | ~13,500 (human-verified, with timestamps and bounding boxes) |
| Trait-level personality analyses | 5,520 (evidence-grounded, one per OCEAN trait per video) |
| Cue-grounding MCQs | 5,320 (seven cognitive categories, six options each) |

## Three Evaluation Tasks

- **Task 1 (Ordinal Personality Rating)**: Predict Big Five trait levels on a 5-point ordinal scale.
- **Task 2 (Open-Ended Rating Reasoning)**: Generate evidence-grounded explanations for each rating, scored by an AI-as-Judge on four dimensions.
- **Task 3 (Structured Cue Grounding)**: Answer multiple-choice questions that probe fine-grained behavioral cue retrieval across seven cognitive categories.

## Four Failure-Mode Metrics

- **PR (Prejudice Rate)**: Correct rating without grounded cues.
- **CR (Confabulation Rate)**: Correct rating with incoherent reasoning.
- **IR (Integration-failure Rate)**: Correct cues but wrong rating.
- **HR (Holistic-Grounding Rate)**: All three tasks correct on the same sample.

## Dataset Structure

```
MM-OCEAN/
├── data/test/          # 1,104 annotation JSONs (one per video)
├── prompts/
│   ├── unified.py      # Three-task evaluation prompt template
│   ├── judge.py        # AI-as-Judge prompt and 5-dimension rubric
│   ├── observer.py     # Observer agent prompt (Stage 1)
│   ├── psychologist.py # Psychologist agent prompt (Stage 2)
│   ├── examiner.py     # Examiner agent prompt (Stage 3)
│   └── aligner.py      # Aligner agent prompt (Stage 4)
├── evaluate.py         # Self-contained scoring script
├── README.md
├── LICENSE
└── croissant.json
```

Each annotation JSON contains:

```json
{
  "video_id": "abc.mp4",
  "transcription": "...",
  "original_scores": {"extraversion": 0.62, ...},
  "observations": [
    {"dimension": "Expression", "start_time": 2.7, "end_time": 4.9,
     "description": "...", "bboxes": [...]}
  ],
  "personality_analyses": [...],
  "mcq_questions": [
    {"category": "Personality Attribution", "question": "...",
     "options": [...], "correct_answer": "C", "explanation": "..."}
  ]
}
```

## Video Access

MM-OCEAN annotations are built on top of **ChaLearn First Impressions V2** videos. Due to licensing, we do not redistribute the videos. To obtain them:

1. Visit the [ChaLearn First Impressions V2](https://chalearnlap.cvc.uab.es/dataset/24/description/) page.
2. Request access and download the test split.
3. Place the `.mp4` files so that `video_id` fields in our JSONs match the filenames.

## Quick Start

### 1. Run your model

Use the prompt template in `prompts/unified.py` to query your model on each video. Save outputs as a JSONL file:

```json
{"video_id": "abc.mp4", "task1": {"extraversion": "High", ...}, "task2": {"extraversion": "The person ...", ...}, "task3": [{"q_idx": 1, "answer": "C"}, ...]}
```

### 2. Run the AI-as-Judge for Task 2

Use the prompt in `prompts/judge.py` to score each Task 2 explanation. Add the judge scores to your JSONL as a `task2_judged` field.

### 3. Evaluate

```bash
python evaluate.py --gt_dir data/test --pred_file results/my_model.jsonl
```

This outputs Task 1 accuracy/MAE, Task 2 composite score, Task 3 overall and per-category accuracy, and the four failure-mode rates (PR/CR/IR/HR).

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

This dataset is released under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The underlying videos follow the ChaLearn First Impressions V2 license. This benchmark is intended for academic research only. See the paper's Appendix (Ethics and Responsible Use) for detailed responsible-use guidelines.
