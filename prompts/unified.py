"""
Unified prompt — All three tasks in a single call per sample.

Task 1: Personality Grading (5-level classification)
Task 2: Personality Reasoning (explanation per trait)
Task 3: Clue Grounding MCQ
"""

SYSTEM = """\
You are an expert in personality psychology trained to assess the Big Five personality traits \
from short video clips of people speaking to camera.
Your judgments must be based solely on behavioral evidence: \
facial expressions, body language, speech patterns, vocal qualities, and verbal content.
Do not make assumptions beyond what is observable."""

USER_TEMPLATE = """\
Watch the following short video of a person being interviewed, and read their transcript.

Transcript:
{transcript}

---

Please complete the following THREE tasks in order. Each task builds on your answers from the previous one.

## Task 1 — Personality Grading

Classify this person on each Big Five trait using a 5-level scale:
  Very Low | Low | Medium | High | Very High

Trait definitions:
- Extraversion:       energy, sociability, assertiveness, talkativeness
- Agreeableness:      cooperativeness, warmth, trust, empathy
- Conscientiousness:  organization, dependability, self-discipline, goal-orientation
- Neuroticism:        emotional instability, anxiety, moodiness, irritability
- Openness:           curiosity, creativity, broad interests, intellectual engagement

## Task 2 — Personality Reasoning

For EACH of the 5 traits, write a 2–3 sentence psychological explanation for the level you assigned in Task 1.
- Cite specific, observable behavioral evidence (facial expressions, gestures, speech rate, vocal tone, verbal content).
- Explain the psychological mechanism linking the behavior to the trait.
- Be concise. Do NOT exceed 3 sentences per trait.

## Task 3 — Clue Grounding MCQ

Answer each multiple-choice question below. Each question asks you to identify the most accurate behavioral evidence supporting a personality conclusion. Avoid selecting hallucinated or fabricated details.

{mcq_block}

---

IMPORTANT: You MUST output a COMPLETE JSON object containing ALL three keys (task1, task2, task3). Do not stop early. Keep each Task 2 explanation to exactly 2 sentences to save space.
For task3, you MUST return exactly {n_questions} entries (q_idx from 1 to {max_q_idx}), one answer per question. Do not omit any q_idx.

Respond ONLY with valid JSON (no markdown fences, no extra text):
{{
  "task1": {{
    "extraversion": "<level>",
    "agreeableness": "<level>",
    "conscientiousness": "<level>",
    "neuroticism": "<level>",
    "openness": "<level>"
  }},
  "task2": {{
    "extraversion": "<2 sentence explanation>",
    "agreeableness": "<2 sentence explanation>",
    "conscientiousness": "<2 sentence explanation>",
    "neuroticism": "<2 sentence explanation>",
    "openness": "<2 sentence explanation>"
  }},
  "task3": [
{task3_template}
  ]
}}"""


def _mcq_block(mcq_questions: list) -> str:
    """Format all MCQ questions for the prompt."""
    parts = []
    for idx, q in enumerate(mcq_questions):
        options_text = "\n".join(f"  {opt['id']}. {opt['text']}" for opt in q.options)
        parts.append(f"Q{idx + 1}: {q.question}\n{options_text}")
    return "\n\n".join(parts)


def _task3_template(mcq_questions: list) -> str:
    """Generate explicit JSON template entries for ALL questions."""
    lines = []
    for idx in range(len(mcq_questions)):
        comma = "," if idx < len(mcq_questions) - 1 else ""
        lines.append(f'    {{"q_idx": {idx + 1}, "answer": "<A/B/C/D/E/F>"}}{comma}')
    return "\n".join(lines)


def build_user_prompt(transcript: str, mcq_questions: list) -> str:
    n_questions = len(mcq_questions)
    return USER_TEMPLATE.format(
        transcript=transcript.strip(),
        mcq_block=_mcq_block(mcq_questions),
        task3_template=_task3_template(mcq_questions),
        n_questions=n_questions,
        max_q_idx=n_questions,
    )


# ── Fallback: Task 3 only (when main call truncates) ─────

TASK3_ONLY_TEMPLATE = """\
Watch the following short video of a person being interviewed, and read their transcript.

Transcript:
{transcript}

---

You previously assessed this person's personality as:
{task1_summary}

Answer each multiple-choice question below with the single correct letter (A/B/C/D/E/F).

{mcq_block}

---

Respond ONLY with valid JSON array (no markdown fences):
[
{task3_template}
]"""


def build_task3_fallback_prompt(
    transcript: str, predictions: dict, mcq_questions: list
) -> str:
    lines = []
    for trait, level in predictions.items():
        lines.append(f"  {trait.title()}: {level}")
    return TASK3_ONLY_TEMPLATE.format(
        transcript=transcript.strip(),
        task1_summary="\n".join(lines),
        mcq_block=_mcq_block(mcq_questions),
        task3_template=_task3_template(mcq_questions),
    )


# ── Fallback: Task 2 missing traits ─────

TASK2_MISSING_TEMPLATE = """\
Watch the following short video of a person being interviewed, and read their transcript.

Transcript:
{transcript}

---

You previously assessed this person's personality as:
{task1_summary}

For each of these traits, write a 2-sentence psychological explanation referencing observable behavior
(facial expressions, body language, speech patterns, vocal qualities, verbal content):
{missing_traits_list}

Respond ONLY with valid JSON (no markdown fences, no extra text), exactly this schema:
{{
{task2_template}
}}"""


def build_task2_fallback_prompt(
    transcript: str, predictions: dict, missing_traits: list
) -> str:
    """Build a fallback prompt that asks ONLY for the missing task2 traits.

    Used when the unified call returned task1+task3 OK but task2 had only a
    subset of the 5 traits (observed for some R-MDPO outputs).
    """
    summary = "\n".join(f"  {t.title()}: {predictions.get(t, 'Medium')}"
                        for t in predictions.keys())
    bullets = "\n".join(f"  - {t}" for t in missing_traits)
    template_lines = []
    for i, t in enumerate(missing_traits):
        comma = "," if i < len(missing_traits) - 1 else ""
        template_lines.append(f'  "{t}": "<2 sentence explanation>"{comma}')
    return TASK2_MISSING_TEMPLATE.format(
        transcript=transcript.strip(),
        task1_summary=summary,
        missing_traits_list=bullets,
        task2_template="\n".join(template_lines),
    )
