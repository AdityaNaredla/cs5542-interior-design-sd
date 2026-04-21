"""
find_best_worst.py

Reads evaluation/results.csv and prints the top/bottom images by CLIP
alignment, plus a few other useful views:
  - Top 5 overall (showcase candidates)
  - Bottom 5 overall (failure case candidates)
  - Best per condition (best baseline, best structured, best controlnet)
  - Worst per condition
  - Rooms where structured MOST improved over baseline (success stories)
  - Rooms where structured FAILED to improve baseline (interesting failures)

Writes all of this to evaluation/rankings.txt for your slides.

Usage:
    python find_best_worst.py
"""

import os
import pandas as pd

RESULTS_CSV = "evaluation/results.csv"
METADATA_CSV = "data/room_metadata.csv"
OUTPUT_TXT = "evaluation/rankings.txt"
N = 5


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"ERROR: {RESULTS_CSV} not found. Run evaluate.py first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    meta = pd.read_csv(METADATA_CSV)
    # Merge scene + style metadata for readable output
    df = df.merge(
        meta[["room_id", "style", "key_objects"]],
        on="room_id", how="left"
    )

    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)

    def row_line(r):
        return (
            f"  {r['room_id']}  {r['condition']:<11s} seed={r['seed']}  "
            f"align={r['clip_alignment']:.4f}  aesth={r['aesthetic_score']:.4f}  "
            f"| {r['scene_category']}, {r['style']}"
        )

    # --- Top N overall ---
    section(f"TOP {N} BY CLIP ALIGNMENT (showcase candidates)")
    top = df.nlargest(N, "clip_alignment")
    for _, r in top.iterrows():
        lines.append(row_line(r))

    # --- Bottom N overall ---
    section(f"BOTTOM {N} BY CLIP ALIGNMENT (failure case candidates)")
    bot = df.nsmallest(N, "clip_alignment")
    for _, r in bot.iterrows():
        lines.append(row_line(r))

    # --- Best/worst per condition ---
    for cond in ("baseline", "structured", "controlnet"):
        sub = df[df["condition"] == cond]
        section(f"BEST & WORST — {cond.upper()}")
        lines.append("  Best:")
        for _, r in sub.nlargest(3, "clip_alignment").iterrows():
            lines.append(row_line(r))
        lines.append("  Worst:")
        for _, r in sub.nsmallest(3, "clip_alignment").iterrows():
            lines.append(row_line(r))

    # --- Where structured most improved over baseline ---
    # Aggregate per room: mean across seeds
    room_cond = df.groupby(["room_id", "condition"])["clip_alignment"].mean().unstack()
    room_cond["struct_gain"] = room_cond["structured"] - room_cond["baseline"]
    room_cond["cn_gain"] = room_cond["controlnet"] - room_cond["baseline"]
    room_cond = room_cond.merge(
        meta[["room_id", "scene_category", "style"]], on="room_id"
    )

    section(f"TOP {N} ROOMS — STRUCTURED IMPROVED MOST OVER BASELINE")
    top_gain = room_cond.nlargest(N, "struct_gain")
    for _, r in top_gain.iterrows():
        lines.append(
            f"  {r['room_id']}  +{r['struct_gain']:.4f}  "
            f"(base={r['baseline']:.3f} → struct={r['structured']:.3f})  "
            f"| {r['scene_category']}, {r['style']}"
        )

    section(f"BOTTOM {N} ROOMS — STRUCTURED DID NOT HELP / HURT BASELINE")
    bot_gain = room_cond.nsmallest(N, "struct_gain")
    for _, r in bot_gain.iterrows():
        lines.append(
            f"  {r['room_id']}  {r['struct_gain']:+.4f}  "
            f"(base={r['baseline']:.3f} → struct={r['structured']:.3f})  "
            f"| {r['scene_category']}, {r['style']}"
        )

    # --- Suggested slide picks ---
    section("SUGGESTED SLIDE PICKS")
    lines.append(
        "Showcase (best comparisons to feature on Results slide):"
    )
    # Rooms where ALL three conditions did well (scene-friendly, good image)
    all_good = room_cond.copy()
    all_good["min_score"] = all_good[["baseline", "structured", "controlnet"]].min(axis=1)
    for _, r in all_good.nlargest(3, "min_score").iterrows():
        lines.append(
            f"  {r['room_id']}  | {r['scene_category']}, {r['style']}  "
            f"(all conditions scored >= {r['min_score']:.3f})"
        )

    lines.append("")
    lines.append("Failure cases (for limitations slide):")
    for _, r in df.nsmallest(3, "clip_alignment").iterrows():
        lines.append(
            f"  {r['room_id']} ({r['condition']})  "
            f"align={r['clip_alignment']:.3f}  "
            f"| {r['scene_category']}, {r['style']}"
        )

    # --- Output ---
    text = "\n".join(lines)
    print(text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved rankings to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()