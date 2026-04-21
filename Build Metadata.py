"""
build_metadata.py
Takes the raw SUN RGB-D CSV (output of extract_sunrgbd_metadata.py) and
produces a balanced, enriched metadata file ready for prompt generation.

Steps:
  1. Filter to high-confidence scene classifications (>= 0.5)
  2. Filter to target scene types with enough real-world visual content
  3. Sample a balanced set (N rooms per scene type)
  4. Enrich each row with style + color_palette + lighting + mood attributes
     (assigned from curated design pools — SUN RGB-D doesn't have these
     and they're essential for stylistic control in generation)

Output: data/room_metadata.csv
"""

import os
import random
import pandas as pd

INPUT_CSV = "data/sun_rgbd_raw.csv"
OUTPUT_CSV = "data/room_metadata.csv"
SAMPLES_PER_SCENE = 3
MIN_CONFIDENCE = 0.5
RANDOM_SEED = 42

# Scenes to include (exclude conference_room - only 27 samples, not enough)
TARGET_SCENES = [
    "bedroom", "living_room", "kitchen", "office",
    "dining_room", "bathroom", "classroom",
]

# Common objects that add noise rather than signal to prompts.
# (E.g., "box" or "garbage_bin" don't help generation.)
NOISE_OBJECTS = {
    "box", "garbage_bin", "bin", "trash", "unknown", "other",
    "object", "thing", "item", "paper", "book", "cup",
}

# Design attribute pools
STYLES = [
    "scandinavian", "modern minimalist", "mid-century modern",
    "industrial loft", "bohemian", "rustic farmhouse",
    "japandi", "contemporary", "art deco",
]

COLOR_PALETTES = [
    "warm neutrals (beige, cream, oak)",
    "cool monochrome (white, grey, charcoal)",
    "earthy tones (terracotta, olive, sand)",
    "navy and brass accents",
    "muted pastels (sage, blush, dove grey)",
    "black and white with walnut wood",
    "forest green with brushed gold",
    "ivory and natural linen",
]

LIGHTING = [
    "soft natural daylight from large windows",
    "warm golden hour sunlight",
    "ambient evening lighting with pendant lamps",
    "bright overcast daylight",
    "layered lighting with floor and table lamps",
    "directional window light with long shadows",
]

MOODS = [
    "cozy and inviting", "clean and airy", "calm and serene",
    "sophisticated and refined", "bright and cheerful",
    "warm and intimate", "sleek and contemporary",
]


def clean_objects(obj_string: str, max_objects: int = 5) -> list:
    """
    Parse semicolon-joined object list, dedupe, remove noise,
    and keep the first N useful objects.
    """
    raw = obj_string.split(";")
    seen = []
    for obj in raw:
        obj = obj.strip().lower().replace("_", " ")
        if not obj or obj in NOISE_OBJECTS or obj.replace(" ", "_") in NOISE_OBJECTS:
            continue
        if obj not in seen:
            seen.append(obj)
        if len(seen) >= max_objects:
            break
    return seen


def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run extract_sunrgbd_metadata.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} raw entries.")

    # Filter by confidence and scene type
    df = df[df["scene_confidence"] >= MIN_CONFIDENCE].copy()
    print(f"After confidence filter (>= {MIN_CONFIDENCE}): {len(df)} entries.")

    df = df[df["scene_category"].isin(TARGET_SCENES)].copy()
    print(f"After scene filter: {len(df)} entries.")

    df = df[df["num_objects"] >= 3].copy()
    print(f"After object-count filter (>= 3 objects): {len(df)} entries.")

    # Sample balanced subset
    sampled = df.groupby("scene_category", group_keys=False).apply(
        lambda g: g.sample(min(len(g), SAMPLES_PER_SCENE), random_state=RANDOM_SEED)
    ).reset_index(drop=True)
    print(f"Sampled to {len(sampled)} balanced entries "
          f"({SAMPLES_PER_SCENE} per scene type).")

    # Enrich each row with design attributes
    enriched_rows = []
    for idx, row in sampled.iterrows():
        key_objects = clean_objects(row["objects"], max_objects=5)
        if len(key_objects) < 2:
            continue  # skip rooms with too little usable content

        enriched_rows.append({
            "room_id": f"room_{idx:03d}",
            "sun_rgbd_image_id": row["image_id"],
            "scene_category": row["scene_category"].replace("_", " "),
            "key_objects": ", ".join(key_objects),
            "style": random.choice(STYLES),
            "color_palette": random.choice(COLOR_PALETTES),
            "lighting": random.choice(LIGHTING),
            "mood": random.choice(MOODS),
            "scene_confidence": row["scene_confidence"],
        })

    out_df = pd.DataFrame(enriched_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved enriched metadata to: {OUTPUT_CSV}")
    print(f"\nPreview (first 5 rows):")
    pd.set_option("display.max_colwidth", 60)
    print(out_df.head(5).to_string(index=False))
    print(f"\nScene distribution:")
    print(out_df["scene_category"].value_counts().to_string())


if __name__ == "__main__":
    main()