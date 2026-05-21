"""
judge.py — LLM-as-Judge prompts for Task 2 evaluation.

Five dimensions, each scored 1-10:
  1. logical_coherence     — behavior → trait logic chain
  2. evidence_coverage     — specific grounded behavioral facts cited
  3. directional_accuracy  — does reasoning support the correct level?
  4. grounding_accuracy    — are cited behaviors actually present in the video?
  5. overall_quality       — holistic assessment
"""

SYSTEM = """\
You are an expert in personality psychology and scientific writing evaluation.
You will be given:
  - A reference explanation (ground-truth annotation, may be in Chinese)
  - A model-generated explanation (in English)
  - The trait being explained
  - The model's predicted level AND the ground-truth (correct) level
  - A list of VERIFIED behavioral observations from the video (human-verified ground truth)

Your task is to score the model explanation on five dimensions (1–10 each):

1. logical_coherence (1-10):
   Does the explanation logically connect specific behavioral observations to the personality trait?
   A score of 10 means the causal chain is clear, psychologically grounded, and free of logical gaps.
   A score of 1 means the reasoning is circular, incoherent, or unsupported.

2. evidence_coverage (1-10):
   Does the explanation cite specific, concrete behavioral evidence from the video/transcript
   (e.g. specific expressions, gestures, speech acts, vocal qualities)?
   A score of 10 means multiple distinct pieces of evidence are precisely identified.
   A score of 1 means no specific evidence is mentioned, only generic assertions.

3. directional_accuracy (1-10):
   Does the explanation support the CORRECT (ground-truth) level, not just the model's predicted level?
   - If the model predicted the correct level and the reasoning supports it: 8-10.
   - If the model predicted wrong but the reasoning contains evidence that actually points toward the correct level: 5-7.
   - If the model predicted wrong AND the reasoning actively argues against the correct level: 1-4.
   This dimension penalizes confident but wrong reasoning.

4. grounding_accuracy (1-10):
   Are the behavioral claims in the model's explanation actually present in the video?
   Compare the model's cited evidence against the VERIFIED OBSERVATIONS list.
   - Score 9-10: All or nearly all cited behaviors match verified observations.
   - Score 6-8: Most cited behaviors are verified, with minor inaccuracies.
   - Score 3-5: Some cited behaviors exist but others are fabricated or significantly wrong
     (e.g., wrong direction: says "looks right" when verified observation says "looks left").
   - Score 1-2: Most cited behaviors appear hallucinated or directly contradict verified observations.
   This is the most critical dimension — it measures whether the model truly PERCEIVED the video
   or merely generated plausible-sounding but ungrounded explanations.

5. overall_quality (1-10):
   Holistic assessment combining all above factors: logical soundness, evidence quality,
   correctness of direction, AND factual grounding.
   A well-written explanation that cites fabricated evidence should score LOW (max 4).
   A less polished explanation that accurately describes real behaviors should score HIGHER.
   A score of 10 is a publishable-quality, factually grounded analysis. A score of 1 is unusable.

Important:
- The VERIFIED OBSERVATIONS are the ground truth for what actually happens in the video.
  If the model claims a behavior that does NOT appear in the verified list, treat it as potentially hallucinated.
- The reference explanation is a quality anchor, not a rubric for string matching.
- Language mismatch (English vs Chinese) does NOT penalize the model.
- Pay special attention to grounding: a confident explanation full of fabricated details is WORSE than
  a cautious explanation that only mentions things actually observable in the video.

Respond ONLY with valid JSON (no markdown):
{{
  "logical_coherence":    <int 1-10>,
  "evidence_coverage":    <int 1-10>,
  "directional_accuracy": <int 1-10>,
  "grounding_accuracy":   <int 1-10>,
  "overall_quality":      <int 1-10>,
  "justification":        "<1-2 sentences explaining your scores, especially noting any hallucinated or fabricated evidence>"
}}"""

USER_TEMPLATE = """\
Trait: {trait_display}
Model's Predicted Level: {level}
Ground-Truth (Correct) Level: {gt_level}

--- Verified Behavioral Observations (human-verified ground truth of what actually happens in the video) ---
{verified_observations}

--- Reference Explanation (ground-truth annotation) ---
{reference}

--- Model Explanation (to be scored) ---
{model_output}"""

TRAIT_DISPLAY = {
    "extraversion":      "Extraversion",
    "agreeableness":     "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "neuroticism":       "Neuroticism",
    "openness":          "Openness",
}


def build_judge_prompt(
    trait: str,
    level: str,
    reference: str,
    model_output: str,
    gt_level: str = "",
    verified_observations: str = "",
) -> str:
    obs_text = verified_observations.strip() if verified_observations else "(No verified observations available)"
    return USER_TEMPLATE.format(
        trait_display=TRAIT_DISPLAY.get(trait, trait.title()),
        level=level,
        gt_level=gt_level or level,
        verified_observations=obs_text,
        reference=reference.strip(),
        model_output=model_output.strip(),
    )
