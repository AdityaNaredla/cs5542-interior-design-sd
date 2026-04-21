"""
prompt_builder.py

Converts structured room metadata into prompts for Stable Diffusion.
Produces THREE prompt variants per row:
  1. naive:      bare-minimum baseline ("a bedroom")
  2. structured: full metadata-derived prompt with style, colors, lighting
  3. negative:   shared negative prompt to steer away from common failure modes

Usage (as module):
    from prompt_builder import build_prompts
    prompts = build_prompts(metadata_row)
    print(prompts["structured"])
"""

# Shared negative prompt - used in BOTH structured and controlnet conditions.
# Calibrated for interior generation failure modes: warped furniture, bad
# perspective, cluttered/messy output, text/watermarks from training data.
NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, warped furniture, "
    "bad perspective, bad anatomy, deformed, disfigured, malformed, "
    "duplicate objects, extra objects, cluttered, messy, dirty, "
    "text, watermark, signature, logo, username, artist name, "
    "cartoon, anime, sketch, painting, illustration, 3d render"
)

# Quality-boosting suffix - appended to structured prompts to nudge
# the model toward photorealistic interior photography aesthetics.
QUALITY_SUFFIX = (
    "interior design photography, architectural digest, "
    "professional lighting, sharp focus, 8k uhd, highly detailed"
)


def build_naive_prompt(row: dict) -> str:
    """Bare-minimum baseline: just the scene type."""
    return f"a {row['scene_category']}"


def build_structured_prompt(row: dict) -> str:
    """
    Full structured prompt combining all metadata fields.
    Template: style + scene + objects + palette + lighting + mood + quality
    """
    return (
        f"A {row['style']} {row['scene_category']}, "
        f"featuring {row['key_objects']}, "
        f"{row['color_palette']} color scheme, "
        f"{row['lighting']}, "
        f"{row['mood']} atmosphere, "
        f"{QUALITY_SUFFIX}"
    )


def build_prompts(row: dict) -> dict:
    """
    Returns a dict with all prompt variants for a given metadata row.
    Compatible with pandas row dicts (use df.to_dict(orient='records')).
    """
    return {
        "naive": build_naive_prompt(row),
        "structured": build_structured_prompt(row),
        "negative": NEGATIVE_PROMPT,
    }


if __name__ == "__main__":
    # Quick self-test: load metadata and preview prompts for row 0
    import pandas as pd

    df = pd.read_csv("data/room_metadata.csv")
    row = df.iloc[0].to_dict()

    prompts = build_prompts(row)
    print(f"Room: {row['room_id']} ({row['scene_category']})")
    print("=" * 60)
    print(f"NAIVE:\n  {prompts['naive']}\n")
    print(f"STRUCTURED:\n  {prompts['structured']}\n")
    print(f"NEGATIVE:\n  {prompts['negative']}")