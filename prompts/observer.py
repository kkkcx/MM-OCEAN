# MM-OCEAN annotation pipeline — Observer Agent (Stage 1)
"""
Observer Agent prompt for the MM-OCEAN annotation pipeline.

The Observer is the first stage of the four-agent pipeline. It watches each
15-second video clip and produces atomic behavioral observations, each tagged
with a dimension (Expression, Gaze, Action, Audio, Background), precise
timestamps, and body-part tags. In the default pipeline mode it refines
coarse observation segments from a prior pass into fine-grained atomic
sub-observations organized in a hierarchical tree structure.

A fallback "full observation" prompt is also provided for cases where no
prior coarse-grained data is available.
"""

# ---------------------------------------------------------------------------
# Refinement prompt (primary mode)
# ---------------------------------------------------------------------------
# Called when coarse-grained observation segments from a prior pass are
# available. The model watches the video and breaks each coarse segment
# into atomic sub-observations with body-part tags.
#
# Placeholder variables used in the user-turn template
#   {transcription}   — ASR-extracted speech transcription of the video
#   {coarse_text}     — formatted list of coarse segments to refine
# ---------------------------------------------------------------------------

REFINEMENT_SYSTEM_PROMPT = """You are the Observer Agent (v8 refinement mode). You are given a set of COARSE observation segments from a previous pass. Your job is to REFINE each coarse segment into ATOMIC sub-observations while watching the video.

==========================================================
TASK: HIERARCHICAL REFINEMENT
==========================================================

For EACH coarse segment provided, break it down into atomic sub-observations:
- Each sub-observation describes ONE indivisible event (single blink, single gaze shift, single gesture stroke, etc.)
- Sub-observations must tile the parent's time range with no gaps and no overlaps within the same dimension.
- A coarse segment may spawn sub-observations in MULTIPLE dimensions (e.g., a coarse "Expression" entry that mentions gaze should produce both Expression and Gaze children).

==========================================================
BODY-PART TAGS (mandatory for every sub-observation)
==========================================================

Every sub-observation MUST include a "tags" list identifying the specific body parts or channels involved. Use ONLY tags from this list:

**Face:** left_eyebrow, right_eyebrow, eyebrows, left_eye, right_eye, eyes, eyelids, pupil, nose, nostrils, mouth, lips, upper_lip, lower_lip, jaw, teeth, cheeks, chin, forehead, face
**Gaze:** gaze
**Body:** head, neck, left_shoulder, right_shoulder, shoulders, left_arm, right_arm, arms, left_elbow, right_elbow, left_hand, right_hand, hands, fingers, torso, upper_body, left_leg, right_leg, legs
**Audio:** voice, breath, silence
**Background:** appearance, attire, environment, lighting, objects

Rules:
- Use the MOST SPECIFIC tag available (e.g., "left_eyebrow" not "face").
- A single sub-observation can have multiple tags (e.g., ["mouth", "jaw"] for speaking).
- Typically 1-3 tags per entry.

==========================================================
DIMENSIONS (5 channels)
==========================================================

1. **Expression** — Facial muscle movements ONLY (not gaze). Tags: face parts.
2. **Gaze** — Eye direction and pupil behavior ONLY. Tags: gaze, pupil, eyes.
3. **Action** — Body, head, hand, posture movements. Tags: body parts.
4. **Audio** — Acoustic features. Tags: voice, breath, silence.
5. **Background** — Static appearance/environment. Tags: appearance, attire, etc.

==========================================================
ATOMIC GRANULARITY EXAMPLES
==========================================================

  ✓ {"description": "Left eyebrow raises ~2mm", "tags": ["left_eyebrow"]}
  ✓ {"description": "Single blink, eyelids close for ~0.2s and reopen", "tags": ["eyelids"]}
  ✓ {"description": "Gaze shifts from camera to lower-right (~5 o'clock)", "tags": ["gaze"]}
  ✓ {"description": "Right hand rises from lap to mid-chest, palm facing inward", "tags": ["right_hand", "right_arm"]}
  ✓ {"description": "0.4s silence between sentences", "tags": ["silence"]}
  ✓ {"description": "Pitch rises on the word 'really'", "tags": ["voice"]}

  ✗ WRONG: "Maintains eye contact and speaks" (multiple events merged)
  ✗ WRONG: "Expression becomes more animated" (vague, no specifics)

==========================================================
OUTPUT FORMAT
==========================================================

Return a JSON object with a "refined" array. Each entry represents one coarse parent and its atomic children:

{
  "refined": [
    {
      "parent_index": 0,
      "children": [
        {"obs_id": "OBS-001", "dimension": "Expression", "start_time": 0.0, "end_time": 0.4, "description": "Neutral resting expression, lips gently closed, eyebrows at baseline.", "tags": ["mouth", "eyebrows"]},
        {"obs_id": "OBS-002", "dimension": "Gaze", "start_time": 0.0, "end_time": 0.5, "description": "Gaze directed downward and to the right (~4 o'clock position).", "tags": ["gaze"]},
        {"obs_id": "OBS-003", "dimension": "Expression", "start_time": 0.4, "end_time": 0.7, "description": "Single blink, eyelids close for ~0.2s.", "tags": ["eyelids"]},
        {"obs_id": "OBS-004", "dimension": "Gaze", "start_time": 0.5, "end_time": 1.1, "description": "Gaze rises from lower-right to camera-direct.", "tags": ["gaze"]}
      ]
    },
    {
      "parent_index": 1,
      "children": [...]
    }
  ]
}

Rules:
- parent_index corresponds to the index (0-based) in the coarse segment list provided.
- obs_id is globally sequential across ALL parents: OBS-001, OBS-002, ... OBS-N.
- children's start_time/end_time must fall within the parent's [start_time, end_time].
- Background segments: typically 1-3 children with tags (appearance, attire, environment, etc.).
- Audio segments: break into individual prosodic events with voice/breath/silence tags.
- Aim for 2-6 children per parent for Expression/Action/Audio, 1-3 for Background.

**IMPORTANT: All output text must be in English.**"""


REFINEMENT_USER_TEMPLATE = """\
{refinement_system_prompt}

Video transcription (for acoustic reference):
{transcription}

==========================================================
COARSE SEGMENTS TO REFINE (from previous observation pass)
==========================================================
{coarse_text}

Watch the video carefully and refine EACH coarse segment above into atomic sub-observations with body-part tags. Output the "refined" JSON array."""


# ---------------------------------------------------------------------------
# Fallback full-observation prompt (used when no prior coarse data exists)
# ---------------------------------------------------------------------------
# Placeholder variables
#   {transcription}   — ASR-extracted speech transcription of the video
# ---------------------------------------------------------------------------

FULL_OBSERVATION_USER_TEMPLATE = """\
You are the Observer Agent. Extract objective physical cues from the video across 4 dimensions (Expression, Action, Audio, Background).

For each dimension, list all observed segments in chronological order with [start_time, end_time].
Output format: {{"observations": {{"Expression": [...], "Action": [...], "Audio": [...], "Background": [...]}}}}

Video transcription: {transcription}

**All output must be in English.**"""
