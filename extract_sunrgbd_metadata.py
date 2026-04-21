"""
extract_sunrgbd_metadata.py

Extracts object annotations from SUNRGBDMeta3DBB_v2.mat and INFERS scene
category from the object list using a rule-based classifier.

Why this approach:
  SUN RGB-D's scene labels live in a separate toolbox file whose schema has
  varied across releases. Rather than fight the .mat parsing, we use the
  object annotations (which ARE cleanly accessible) to classify scenes.
  This is a transparent, documented methodology - scene inference from
  object composition is a standard technique in indoor scene understanding.

Produces: data/sun_rgbd_raw.csv with columns:
  image_id, scene_category, objects, num_objects, scene_confidence
"""

import os
import pandas as pd
from scipy.io import loadmat
from collections import Counter

BBOX_MAT = "data/sun_rgbd_raw/SUNRGBDMeta3DBB_v2.mat"
OUTPUT_CSV = "data/sun_rgbd_raw.csv"


# Object → scene mapping. Objects mapped to multiple scenes are handled
# via weighted voting (an object can weakly suggest multiple rooms).
# Keys are normalized lowercase (we also handle common aliases).
SCENE_INDICATORS = {
    "bedroom": {
        "bed": 5, "pillow": 3, "nightstand": 4, "night_stand": 4, "dresser": 3,
        "wardrobe": 3, "bedside": 4, "headboard": 4, "blanket": 2,
    },
    "kitchen": {
        "stove": 5, "refrigerator": 5, "fridge": 5, "microwave": 4, "oven": 4,
        "sink": 2, "cabinet": 1, "counter": 2, "kitchen_cabinet": 5,
        "dishwasher": 5, "cutting_board": 4, "pot": 3, "pan": 3,
    },
    "bathroom": {
        "toilet": 5, "bathtub": 5, "shower": 5, "sink": 3, "towel": 3,
        "mirror": 1, "bath": 5,
    },
    "living_room": {
        "sofa": 5, "couch": 5, "coffee_table": 4, "tv": 3, "television": 3,
        "ottoman": 3, "armchair": 4, "entertainment_center": 4,
    },
    "dining_room": {
        "dining_table": 5, "chair": 1, "tablecloth": 2, "plate": 2,
    },
    "office": {
        "desk": 5, "computer": 3, "monitor": 3, "keyboard": 3, "office_chair": 5,
        "printer": 3, "file_cabinet": 4, "filing_cabinet": 4, "laptop": 3,
    },
    "classroom": {
        "whiteboard": 5, "blackboard": 5, "chalkboard": 5, "school_desk": 5,
        "podium": 4, "lecture": 4,
    },
    "conference_room": {
        "conference_table": 5, "projector": 3, "screen": 2, "whiteboard": 3,
    },
}


def safe_str(val) -> str:
    if val is None:
        return ""
    try:
        s = str(val).strip().lower()
        if s in ("nan", "none", "[]"):
            return ""
        return s
    except Exception:
        return ""


def extract_objects(entry) -> list:
    objects = []
    bbs = getattr(entry, "groundtruth3DBB", None)
    if bbs is None:
        return objects
    if hasattr(bbs, "_fieldnames"):
        bbs = [bbs]
    try:
        bbs_iter = iter(bbs)
    except TypeError:
        return objects
    for bb in bbs_iter:
        cn = safe_str(getattr(bb, "classname", None))
        if cn:
            # Normalize: replace spaces with underscores
            cn = cn.replace(" ", "_")
            objects.append(cn)
    return objects


def infer_scene(objects: list) -> tuple:
    """
    Returns (scene_category, confidence_score).
    Confidence is the winning scene's score divided by total score.
    Returns (None, 0.0) if no clear winner.
    """
    scores = {scene: 0 for scene in SCENE_INDICATORS}
    for obj in objects:
        for scene, indicators in SCENE_INDICATORS.items():
            if obj in indicators:
                scores[scene] += indicators[obj]

    total_score = sum(scores.values())
    if total_score == 0:
        return (None, 0.0)

    winner = max(scores, key=scores.get)
    if scores[winner] == 0:
        return (None, 0.0)

    confidence = scores[winner] / total_score
    return (winner, confidence)


def main():
    if not os.path.exists(BBOX_MAT):
        print(f"ERROR: {BBOX_MAT} not found.")
        return

    print(f"Loading {BBOX_MAT}...")
    mat = loadmat(BBOX_MAT, struct_as_record=False, squeeze_me=True)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    meta = mat[keys[0]]
    print(f"  {keys[0]}: {len(meta)} entries")

    print("\nExtracting objects and inferring scenes...")
    rows = []
    no_objects = 0
    unclassified = 0

    for entry in meta:
        seq = safe_str(getattr(entry, "sequenceName", None))
        if not seq:
            continue
        objects = extract_objects(entry)
        if not objects:
            no_objects += 1
            continue

        scene, confidence = infer_scene(objects)
        if scene is None:
            unclassified += 1
            continue

        rows.append({
            "image_id": seq,
            "scene_category": scene,
            "objects": ";".join(objects),
            "num_objects": len(objects),
            "scene_confidence": round(confidence, 3),
        })

    print(f"  Classified:             {len(rows)}")
    print(f"  Skipped (no objects):   {no_objects}")
    print(f"  Skipped (unclassified): {unclassified}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total classified: {len(df)}")

    if len(df) == 0:
        return

    print(f"\nScene distribution:")
    for scene, count in df["scene_category"].value_counts().items():
        print(f"  {scene:20s} {count:5d}")

    print(f"\nConfidence statistics:")
    print(f"  mean: {df['scene_confidence'].mean():.3f}")
    print(f"  high-confidence (>0.5):   {(df['scene_confidence'] > 0.5).sum()}")
    print(f"  medium (0.3-0.5):         {((df['scene_confidence'] > 0.3) & (df['scene_confidence'] <= 0.5)).sum()}")
    print(f"  low (<0.3):               {(df['scene_confidence'] <= 0.3).sum()}")

    all_objects = []
    for obj_str in df["objects"]:
        all_objects.extend(obj_str.split(";"))
    print(f"\nUnique object classes in extracted data: {len(set(all_objects))}")
    print(f"Top 15 objects:")
    for obj, count in Counter(all_objects).most_common(15):
        print(f"  {obj:30s} {count:5d}")


if __name__ == "__main__":
    main()