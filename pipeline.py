"""
pipeline.py

Main generation pipeline for the Interior Design Generation challenge.
Runs three conditions per room in the metadata CSV:

  1. baseline    — naive prompt, no negative, no control
  2. structured  — structured prompt + negative prompt
  3. controlnet  — structured prompt + negative prompt + MLSD line control

ControlNet workflow:
  Since we generate from scratch (not from photos), we use the baseline
  image's MLSD line map as the layout control for the controlnet condition.
  This tests whether preserving structural lines (walls, furniture edges)
  while improving prompts yields better results than prompting alone.

Outputs: outputs/{condition}/{room_id}_seed{seed}.png

Usage:
    python pipeline.py                      # run all rooms, all conditions
    python pipeline.py --test               # run first 2 rooms only (smoke test)
    python pipeline.py --skip baseline      # skip a condition
"""

import torch
import os
import argparse
import gc
import time
import pandas as pd
from PIL import Image


from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from controlnet_aux import MLSDdetector

from prompt_builder import build_prompts

# --- Config ---
METADATA_CSV = "data/room_metadata.csv"
OUTPUT_DIR = "outputs"
SD_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-mlsd"
MLSD_DETECTOR_MODEL = "lllyasviel/Annotators"
IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 0.8
SEEDS = [42, 1337]  # two seeds per condition for diversity measurements
DEVICE = "cuda"


def make_output_dirs():
    for cond in ("baseline", "structured", "controlnet", "control_maps"):
        os.makedirs(os.path.join(OUTPUT_DIR, cond), exist_ok=True)


def load_sd_pipeline() -> StableDiffusionPipeline:
    print("Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL, torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


def load_controlnet_pipeline() -> StableDiffusionControlNetPipeline:
    print("Loading ControlNet-MLSD pipeline...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


def generate_one(pipe, prompt: str, negative: str, seed: int,
                 control_image=None) -> Image.Image:
    """Single generation - handles both SD and ControlNet pipelines."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
    )
    if control_image is not None:
        kwargs["image"] = control_image
        kwargs["controlnet_conditioning_scale"] = CONTROLNET_SCALE
    return pipe(**kwargs).images[0]


def run_conditions_for_row(row: dict, conditions: list,
                           sd_pipe, cn_pipe, mlsd_detector):
    """Generate images for one room across all requested conditions."""
    prompts = build_prompts(row)
    room_id = row["room_id"]

    baseline_image = None  # need this to build the control map

    for seed in SEEDS:
        # --- baseline ---
        if "baseline" in conditions:
            img = generate_one(
                sd_pipe, prompts["naive"], negative="", seed=seed
            )
            out_path = os.path.join(
                OUTPUT_DIR, "baseline", f"{room_id}_seed{seed}.png"
            )
            img.save(out_path)
            print(f"  [baseline] saved {out_path}")
            if baseline_image is None:
                baseline_image = img  # reuse first baseline as layout source

        # --- structured ---
        if "structured" in conditions:
            img = generate_one(
                sd_pipe, prompts["structured"],
                negative=prompts["negative"], seed=seed
            )
            out_path = os.path.join(
                OUTPUT_DIR, "structured", f"{room_id}_seed{seed}.png"
            )
            img.save(out_path)
            print(f"  [structured] saved {out_path}")

        # --- controlnet ---
        if "controlnet" in conditions:
            if baseline_image is None:
                # If baseline was skipped, make one to extract lines from
                baseline_image = generate_one(
                    sd_pipe, prompts["naive"], negative="", seed=SEEDS[0]
                )
            control_map = mlsd_detector(baseline_image)
            # Save the control map once per room (for slides)
            cm_path = os.path.join(
                OUTPUT_DIR, "control_maps", f"{room_id}_mlsd.png"
            )
            if not os.path.exists(cm_path):
                control_map.save(cm_path)

            img = generate_one(
                cn_pipe, prompts["structured"],
                negative=prompts["negative"], seed=seed,
                control_image=control_map
            )
            out_path = os.path.join(
                OUTPUT_DIR, "controlnet", f"{room_id}_seed{seed}.png"
            )
            img.save(out_path)
            print(f"  [controlnet] saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="run first 2 rooms only (smoke test)")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["baseline", "structured", "controlnet"],
                        help="conditions to skip")
    args = parser.parse_args()

    make_output_dirs()

    df = pd.read_csv(METADATA_CSV)
    if args.test:
        df = df.head(2)
    print(f"Running pipeline on {len(df)} rooms.")

    conditions = ["baseline", "structured", "controlnet"]
    conditions = [c for c in conditions if c not in args.skip]
    print(f"Active conditions: {conditions}")

    # --- Load models ---
    sd_pipe = load_sd_pipeline() if (
        "baseline" in conditions or "structured" in conditions
    ) else None
    cn_pipe = load_controlnet_pipeline() if "controlnet" in conditions else None
    mlsd_detector = (
        MLSDdetector.from_pretrained(MLSD_DETECTOR_MODEL)
        if "controlnet" in conditions else None
    )

    # --- Run ---
    t0 = time.time()
    for i, row in enumerate(df.to_dict(orient="records")):
        print(f"\n[{i+1}/{len(df)}] {row['room_id']} "
              f"({row['scene_category']}, {row['style']})")
        run_conditions_for_row(row, conditions, sd_pipe, cn_pipe, mlsd_detector)

        # Free memory between rooms
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()