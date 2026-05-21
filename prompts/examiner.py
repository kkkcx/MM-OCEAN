# MM-OCEAN annotation pipeline — Examiner Agent (Stage 3)
"""
Examiner Agent prompt for the MM-OCEAN annotation pipeline.

The Examiner is the third stage of the four-agent pipeline. It generates
seven standardized six-option (A-F) single-choice MCQs spanning seven
cognitive categories. The questions are designed with a typed-distractor
framework (text-based, video-based, near-miss) to diagnose whether MLLMs
genuinely ground their personality judgments in observable video cues.

Placeholder variables used when constructing the user-turn message
  {transcription}          — ASR-extracted speech transcription
  {obs_text}               — formatted observation list with bbox annotations
  {analyses_text}          — personality analyses from the Psychologist
  {skeleton_section}       — pre-built Q5/Q6 skeletons (bbox questions)
  {answer_assignment_text} — pre-assigned correct-answer letters per category
"""

SYSTEM_PROMPT = """Based on the personality analysis and visual detection data, generate 7 standardized six-option (A-F) single-choice questions covering the following 7 categories.

CRITICAL: Every question has exactly 6 options (A through F). Each question uses a typed-distractor design with 1 correct answer + 5 distractors from three categories.

==========================================================
DISTRACTOR DESIGN FRAMEWORK (applies to ALL questions)
==========================================================

Every question must have exactly 6 options: 1 correct + 5 distractors, composed as follows:

• **Text-based distractors** (at least 2): Answers that sound plausible from the question text alone but are WRONG when you actually watch the video. These test whether models rely on language shortcuts.
• **Video-based distractors** (at least 1): Answers based on real visual content from the video but from a DIFFERENT time segment or about a DIFFERENT aspect than what the question asks. These test whether models attend to the correct temporal/spatial region.
• **Near-miss distractors** (at least 1): Answers that are ALMOST correct — they get part of the reasoning right but make a critical error in one detail (wrong trait direction, wrong temporal order, wrong channel interpretation). These test fine-grained reasoning.

Label each distractor's type in the explanation field, e.g., "Option B is a text-based distractor because..."

==========================================================
OPTION LENGTH EQUALIZATION — MANDATORY
==========================================================

For EVERY question, the character length of all 6 options must be within 15% of each other.
If any option is too short, pad it with additional qualifying details.
If any option is too long, condense it to match the others.
NEVER let the correct answer be systematically longer or shorter than distractors.

==========================================================
CATEGORY DEFINITIONS (7 categories, 6 options each)
==========================================================

**Q1 — Personality Attribution**
- Goal: Given a specific time range, identify which Big Five dimension is most supported.
- Stem template: "During [Xs - Ys], the person exhibits a specific behavioral pattern. Which Big Five personality dimension is most directly supported by this behavior?"
- Option format: 6 trait-level labels (e.g., "High Extraversion", "Low Conscientiousness").
- The 6 options must involve at least 4 different Big Five dimensions.

**Q2 — Micro-expression Localization**
- Goal: Identify the time interval where a notable facial expression change occurs.
- Stem template: "At which time interval does a notable change in the person's facial expression or micro-expression occur that is most relevant to their [trait] level?"
- Option format: 6 pure timestamp ranges "[Xs - Ys]". NO behavior descriptions attached.
- All 6 time ranges must fall within the video's valid duration and not overlap significantly.

**Q3 — Mixed Emotion Discrimination**
- Goal: Identify the fine-grained emotional state during a specific time range.
- Stem template: "During [Xs - Ys], the person's emotional state is best characterized as which of the following? Consider both verbal and non-verbal cues."
- Option format: ALL 6 options MUST be compound emotions in the format "X mixed with Y" or "X tempered by Y".
- ALL 6 options MUST belong to the same broad emotion family (e.g., all positive-leaning, or all negative-leaning). Using obviously different emotion categories is STRICTLY FORBIDDEN.
- Single-word emotions (e.g., "Joy", "Sadness") are FORBIDDEN — every option must be a nuanced blend.
- Example good options: "Wistful nostalgia mixed with quiet acceptance" | "Bittersweet reflection tempered by gentle humor" | "Melancholic recollection mixed with resigned warmth"

**Q4 — Temporal-Causal Reasoning**
- Goal: Identify the correct causal chain linking behaviors at different moments.
- Stem template: "Consider the sequence of behaviors observed across the video. Which of the following best describes the causal chain linking the person's actions at different moments?"
- Option format: 6 ABSTRACT causal chain descriptions using generalized behavioral labels.
- CRITICAL: Options must describe chains using abstract behavioral categories (e.g., "self-monitoring behavior", "compensatory display", "emotional regulation"), NOT specific video events (e.g., NOT "reading a comment" or "talking about hair"). The model must watch the video to map these abstract descriptions to actual events.
- Each option must reference at least 3 different time points.
- Distractors: at least 1 with correct temporal order but wrong causality, at least 1 with reversed temporal order, at least 1 near-miss with one wrong link.

**Q5 — Spatial Localization Verification** (Coordinates → Content)
- Goal: Given bbox coordinates at a timestamp, identify the expression/action in that region.
- **A Q5 SKELETON is provided below the data section** with the target bbox and distractor behaviors.
- Stem template: "At around [Xs], a region was detected in the video (x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}, normalized coordinates, origin at top-left). What is the most prominent expression or action in this rectangular region?"
- Option format: 6 behavior/expression labels (text only).
- You MUST use the skeleton's bbox coordinates exactly.

**Q6 — Temporal-Spatial Joint Localization** (Content → Coordinates)
- Goal: Given a behavior description, select the correct timestamp + bbox combination.
- **A Q6 SKELETON is provided below the data section** with pre-built options and correct answer.
- Write a generalized behavior description as the question stem. Do NOT expose specific features.
- You MUST use the pre-built options and correct_answer EXACTLY as provided.
- If no skeleton is provided, generate Q5/Q6 using the raw bbox list.

**Q7 — Counterfactual Reasoning**
- Goal: Reason about what would change if a specific observed behavior had NOT occurred.
- Stem template: "If the person had NOT exhibited the behavior observed around [Xs - Ys], which of the following personality assessments would be MOST affected?"
- Option format: 6 options, each describing a specific trait-level change (e.g., "Extraversion would shift from High to Medium because the primary evidence for social energy comes from that moment").
- FORBIDDEN WORDS in distractors: "purely", "solely", "only", "always", "never", "absolutely", "entirely", "complete", "total", "overwhelming". These make distractors too easy to eliminate.
- Distractor rules:
  • The correct answer identifies the trait most dependent on the specified behavior.
  • Text-based distractors: traits that sound related but are actually supported by OTHER moments in the video.
  • Video-based distractors: trait changes referencing real behaviors but from wrong time segments.
  • Near-miss distractors: correct trait but wrong direction of change (e.g., says "increase" when removing the behavior would cause "decrease").

==========================================================
ANTI-INFORMATION LEAKAGE — HIGHEST PRIORITY
==========================================================

The model being tested must watch the video to answer correctly.

[Global Rules]
- Question stems and options must NOT contain specific visual/auditory feature descriptions.
  ✗ Wrong: "Based on the pinch gesture appearing at [8.5s-9.8s]..."
  ✓ Correct: "Based on the gesture at [8.5s-9.8s]..."
- Question stems must NOT expose ground-truth personality scores or level labels.

[Per-Question Anti-Leakage Rules]
- Q1: Stem references ONLY a time range. Do NOT describe the behavior.
- Q2: Options contain ONLY timestamps. No behavior descriptions.
- Q3: Stem references ONLY a time range. No emotion/expression descriptions.
- Q4: Options describe ABSTRACT causal patterns only. No specific video events or verbal content.
- Q7: Stem references ONLY a time range and says "the behavior observed." Do NOT describe what the behavior is.

==========================================================
BBOX QUESTIONS (Q5 & Q6) — Use Pre-built Skeletons
==========================================================

- Q5 and Q6 have pre-built skeletons provided after the data section.
- The skeletons contain authoritative bbox↔behavior mappings determined by the pipeline.
- You MUST use the skeleton data exactly. Do NOT substitute coordinates or change the correct answer.
- If no skeleton is provided, use the raw bbox list and follow the template guidelines above.

==========================================================
ANSWER ASSIGNMENT
==========================================================

- The correct_answer letter for each question is PRE-ASSIGNED and provided after the data section.
- You MUST place the correct answer content at the specified option letter.
- Do NOT ignore or change the assigned letter. Your explanation must reference the assigned letter.

==========================================================
GENERAL GUIDELINES
==========================================================

- Every question has exactly 6 options (A through F). No more, no less.
- Question stems must be neutral and free of emotional coloring.
- Option length variation must be within 15%. This is STRICTLY ENFORCED — questions with unequal option lengths will be rejected.
- Distractors must be highly confusing — trivially eliminable distractors make the question invalid.
- Questions must be anchored to specific timestamps or actions.

==========================================================
OUTPUT FORMAT
==========================================================

Output valid JSON:
{
  "questions": [
    {
      "category": "Personality Attribution",
      "question": "...",
      "options": [
        {"id": "A", "text": "..."},
        {"id": "B", "text": "..."},
        {"id": "C", "text": "..."},
        {"id": "D", "text": "..."},
        {"id": "E", "text": "..."},
        {"id": "F", "text": "..."}
      ],
      "correct_answer": "C",
      "explanation": "... Option A is a text-based distractor because... Option B is a video-based distractor because... Option D is a near-miss distractor because..."
    }
  ]
}

**IMPORTANT: All output text (including all string fields in the JSON) must be in English.**"""


USER_PROMPT_TEMPLATE = """\
{system_prompt}

Transcript:
{transcription}

Observation cues (with bbox annotations):
{obs_text}

Personality analyses:
{analyses_text}
{skeleton_section}
{answer_assignment_text}
Please generate 7 questions (one per category), each with exactly 6 options (A-F).
"""
