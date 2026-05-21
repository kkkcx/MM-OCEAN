# MM-OCEAN annotation pipeline — Aligner Agent (Stage 4)
"""
Aligner Agent prompt for the MM-OCEAN annotation pipeline.

The Aligner is the fourth and final stage of the four-agent pipeline. It
performs quality assurance on the generated MCQ questions through two
complementary mechanisms.

1. Deterministic code-based pre-validation. The pipeline code checks
   timestamp ranges against video duration, validates bounding-box
   coordinate legality, and verifies bbox existence before the LLM is
   called. Issues found here are surfaced as hints in the LLM prompt.

2. LLM-based semantic review (without re-uploading the video). The LLM
   checks semantic consistency between correct answers and personality
   analyses, and factual accuracy of timestamp/bbox references against
   the observation records. It returns corrections that the pipeline
   applies automatically.

Placeholder variables used when constructing the user-turn message
  {compact_data}   — JSON blob containing video_id, transcription,
                     personality_analyses, relevant_observations, and
                     mcq_questions (only observations referenced by the
                     MCQs are included to reduce token cost)
  {code_hint}      — optional section listing issues found by the
                     deterministic code pre-validation step
"""

SYSTEM_PROMPT = """You are a quality-control expert for personality MCQ datasets. Review the questions below and check for TWO things only:

**Check A — Semantic Consistency:**
Is the correct answer to each question logically consistent with the personality analyses provided? If the analysis says a trait is "High" but the correct answer contradicts this, flag it.

**Check B — Factual Accuracy:**
For questions that reference specific timestamps or bounding-box coordinates:
- Does the described behavior actually match what the observation records say happened at that time?
- Are the correct answer and distractors properly differentiated?
- For bbox questions (Spatial Localization / Temporal-Spatial Joint): does the correct option's bbox genuinely correspond to the behavior described?

Rules:
- Only fix question stems, options, correct_answer, or explanations.
- If no issues are found, return {"validated": true, "quality_issues": [], "corrections": {}}.
- Be concise. Only flag genuine errors, not stylistic preferences.

Correction key format:
- mcq_questions[N].correct_answer      → change correct answer letter
- mcq_questions[N].question            → change question stem
- mcq_questions[N].options             → replace all options (list of {id, text})
- mcq_questions[N].options[M].text     → change one option's text
- mcq_questions[N].explanation         → change explanation text

Output ONLY valid JSON:
{
  "validated": true/false,
  "quality_issues": [
    {"check": "A or B", "q_idx": 0, "issue": "...", "action": "..."}
  ],
  "corrections": {}
}

**All output must be in English.**"""


USER_PROMPT_TEMPLATE = """\
{system_prompt}
{code_hint}
Data to review:
{compact_data}
"""
