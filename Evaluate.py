import torch  # MUST be first on Windows — DLL load order

import os
import itertools
import pandas as pd
import numpy as np
from PIL import Image
import open_clip
import lpips
from tqdm import tqdm

from prompt_builder import build_prompts

# --- Config ---
METADATA_CSV = "data/room_metadata.csv"
OUTPUT_DIR = "outputs"
EVAL_DIR = "evaluation"
CONDITIONS = ["baseline", "structured", "controlnet"]
SEEDS = [42, 1337]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reference prompt for aesthetic proxy score — all images scored against
# this; higher = more "aesthetically interior-design-y"
AESTHETIC_REF = (
    "professional interior design photography, beautiful, high quality, "
    "magazine editorial, sharp focus, well-composed"
)


def load_clip():
    print("Loading CLIP (ViT-B/32)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(DEVICE).eval()
    return model, preprocess, tokenizer


def load_lpips():
    print("Loading LPIPS (AlexNet backbone)...")
    return lpips.LPIPS(net="alex").to(DEVICE).eval()


@torch.no_grad()
def clip_score(model, preprocess, tokenizer, image_path: str, text: str) -> float:
    """Cosine similarity between image and text embeddings, in [-1, 1]."""
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    tokens = tokenizer([text]).to(DEVICE)

    img_feat = model.encode_image(img)
    txt_feat = model.encode_text(tokens)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    return (img_feat @ txt_feat.T).item()


@torch.no_grad()
def lpips_distance(lpips_model, img_path_a: str, img_path_b: str) -> float:
    """Perceptual distance in [0, ~1]. Higher = more different."""
    def to_tensor(path):
        img = Image.open(path).convert("RGB").resize((256, 256))
        arr = np.array(img).astype(np.float32) / 255.0  # [0, 1]
        arr = arr * 2.0 - 1.0  # LPIPS expects [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    return lpips_model(to_tensor(img_path_a), to_tensor(img_path_b)).item()


def get_image_path(condition: str, room_id: str, seed: int) -> str:
    return os.path.join(OUTPUT_DIR, condition, f"{room_id}_seed{seed}.png")


def evaluate_all():
    os.makedirs(EVAL_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_CSV)
    clip_model, clip_preprocess, clip_tokenizer = load_clip()
    lpips_model = load_lpips()

    # --- Per-image alignment + aesthetic scores ---
    per_image_rows = []
    print("\nScoring all images (alignment + aesthetic)...")
    for row in tqdm(df.to_dict(orient="records"), desc="rooms"):
        prompts = build_prompts(row)
        # Alignment is measured against the STRUCTURED prompt — it's the
        # full description of the intended image. This is the fair
        # comparison: baseline should do worse because it was generated
        # from less info, even though measured against the same target.
        target_prompt = prompts["structured"]

        for condition in CONDITIONS:
            for seed in SEEDS:
                img_path = get_image_path(condition, row["room_id"], seed)
                if not os.path.exists(img_path):
                    continue

                align = clip_score(
                    clip_model, clip_preprocess, clip_tokenizer,
                    img_path, target_prompt
                )
                aesth = clip_score(
                    clip_model, clip_preprocess, clip_tokenizer,
                    img_path, AESTHETIC_REF
                )
                per_image_rows.append({
                    "room_id": row["room_id"],
                    "scene_category": row["scene_category"],
                    "condition": condition,
                    "seed": seed,
                    "clip_alignment": align,
                    "aesthetic_score": aesth,
                })

    per_image_df = pd.DataFrame(per_image_rows)

    # --- Diversity: LPIPS between seed variants within same condition ---
    print("\nComputing LPIPS diversity (pairs of seeds per condition)...")
    diversity_rows = []
    for row in tqdm(df.to_dict(orient="records"), desc="rooms"):
        for condition in CONDITIONS:
            paths = [get_image_path(condition, row["room_id"], s) for s in SEEDS]
            paths = [p for p in paths if os.path.exists(p)]
            if len(paths) < 2:
                continue
            distances = []
            for a, b in itertools.combinations(paths, 2):
                distances.append(lpips_distance(lpips_model, a, b))
            diversity_rows.append({
                "room_id": row["room_id"],
                "condition": condition,
                "lpips_diversity": float(np.mean(distances)),
            })

    diversity_df = pd.DataFrame(diversity_rows)

    # --- Save per-image results ---
    full_df = per_image_df.merge(
        diversity_df, on=["room_id", "condition"], how="left"
    )
    full_df.to_csv(os.path.join(EVAL_DIR, "results.csv"), index=False)
    print(f"\nSaved per-image results: {EVAL_DIR}/results.csv")

    # --- Summary table (what goes on the slide) ---
    summary = per_image_df.groupby("condition").agg(
        clip_alignment_mean=("clip_alignment", "mean"),
        clip_alignment_std=("clip_alignment", "std"),
        aesthetic_mean=("aesthetic_score", "mean"),
        aesthetic_std=("aesthetic_score", "std"),
    ).round(4)

    div_summary = diversity_df.groupby("condition").agg(
        lpips_diversity_mean=("lpips_diversity", "mean"),
        lpips_diversity_std=("lpips_diversity", "std"),
    ).round(4)

    summary = summary.join(div_summary)
    summary = summary.reindex(CONDITIONS)  # force ordering
    summary.to_csv(os.path.join(EVAL_DIR, "summary.csv"))

    print("\n" + "=" * 70)
    print("SUMMARY (per condition)")
    print("=" * 70)
    print(summary.to_string())

    # --- Human-readable report ---
    with open(os.path.join(EVAL_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("CS 5542 — Interior Design Generation: Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write("Rooms evaluated: " + str(len(df)) + "\n")
        f.write("Conditions: " + ", ".join(CONDITIONS) + "\n")
        f.write("Seeds per condition: " + str(len(SEEDS)) + "\n\n")
        f.write("Per-condition means:\n")
        f.write(summary.to_string())
        f.write("\n\n")

        # Simple delta analysis vs baseline
        f.write("Deltas vs baseline:\n")
        base = summary.loc["baseline"]
        for cond in CONDITIONS:
            if cond == "baseline":
                continue
            d_align = summary.loc[cond, "clip_alignment_mean"] - base["clip_alignment_mean"]
            d_aesth = summary.loc[cond, "aesthetic_mean"] - base["aesthetic_mean"]
            d_div = summary.loc[cond, "lpips_diversity_mean"] - base["lpips_diversity_mean"]
            f.write(f"  {cond}:\n")
            f.write(f"    alignment delta:   {d_align:+.4f}\n")
            f.write(f"    aesthetic delta:   {d_aesth:+.4f}\n")
            f.write(f"    diversity delta:   {d_div:+.4f}\n")

    print(f"\nSaved human-readable summary: {EVAL_DIR}/summary.txt")
    print("\nDone.")


if __name__ == "__main__":
    evaluate_all()