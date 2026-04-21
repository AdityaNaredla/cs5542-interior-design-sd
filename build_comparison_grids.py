"""
build_comparison_grids.py

Creates one 4-panel comparison grid per room showing:
    [ baseline | structured | controlnet | MLSD control map ]

Each panel is labeled. Saved to outputs/grids/ as {room_id}_grid.png.
Uses seed 42 images by default (the "canonical" seed for each room).

Usage:
    python build_comparison_grids.py
    python build_comparison_grids.py --seed 1337   # use the other seed
"""

import os
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- Config ---
METADATA_CSV = "data/room_metadata.csv"
OUTPUT_DIR = "outputs"
GRID_DIR = "outputs/grids"
PANEL_SIZE = 384       # each panel in the grid
LABEL_HEIGHT = 40
HEADER_HEIGHT = 50
PANEL_GAP = 8
BG = (245, 245, 245)
LABEL_BG = (30, 30, 30)
LABEL_FG = (255, 255, 255)
HEADER_FG = (20, 20, 20)


def load_font(size: int):
    """Try a few common Windows/mac/linux fonts, fall back to default."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def load_or_placeholder(path: str, size: int) -> Image.Image:
    """Load image, or return a grey placeholder with 'MISSING' text."""
    if os.path.exists(path):
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (size, size), (200, 200, 200))
        d = ImageDraw.Draw(img)
        d.text((size // 2 - 40, size // 2 - 10), "MISSING",
               fill=(80, 80, 80), font=load_font(18))
    return img.resize((size, size), Image.LANCZOS)


def panel_with_label(image: Image.Image, label: str) -> Image.Image:
    """Adds a label bar above the image panel."""
    w, h = image.size
    out = Image.new("RGB", (w, h + LABEL_HEIGHT), LABEL_BG)
    out.paste(image, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(out)
    font = load_font(18)
    # Center text in label bar
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:  # older Pillow
        text_w, text_h = draw.textsize(label, font=font)
    x = (w - text_w) // 2
    y = (LABEL_HEIGHT - text_h) // 2
    draw.text((x, y), label, fill=LABEL_FG, font=font)
    return out


def build_grid_for_room(row: dict, seed: int) -> Image.Image:
    room_id = row["room_id"]

    # Load all four panels
    panels = []
    for condition, subdir in [
        ("Baseline", "baseline"),
        ("Structured", "structured"),
        ("ControlNet (MLSD)", "controlnet"),
    ]:
        img_path = os.path.join(OUTPUT_DIR, subdir, f"{room_id}_seed{seed}.png")
        img = load_or_placeholder(img_path, PANEL_SIZE)
        panels.append(panel_with_label(img, condition))

    # Control map (shared across seeds)
    cm_path = os.path.join(OUTPUT_DIR, "control_maps", f"{room_id}_mlsd.png")
    cm_img = load_or_placeholder(cm_path, PANEL_SIZE)
    panels.append(panel_with_label(cm_img, "MLSD Control Map"))

    # Assemble grid
    labeled_h = PANEL_SIZE + LABEL_HEIGHT
    total_w = PANEL_SIZE * 4 + PANEL_GAP * 3
    total_h = HEADER_HEIGHT + labeled_h

    grid = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(grid)

    # Header text: room id + scene + style
    header = (f"{room_id}  |  {row['scene_category'].title()}  |  "
              f"{row['style']}  |  seed={seed}")
    font = load_font(22)
    draw.text((12, 12), header, fill=HEADER_FG, font=font)

    # Paste panels
    x = 0
    for panel in panels:
        grid.paste(panel, (x, HEADER_HEIGHT))
        x += PANEL_SIZE + PANEL_GAP

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="which seed's images to use in the grid")
    args = parser.parse_args()

    os.makedirs(GRID_DIR, exist_ok=True)
    df = pd.read_csv(METADATA_CSV)

    print(f"Building {len(df)} comparison grids (seed={args.seed})...")
    for row in df.to_dict(orient="records"):
        grid = build_grid_for_room(row, args.seed)
        out_path = os.path.join(GRID_DIR, f"{row['room_id']}_grid.png")
        grid.save(out_path, quality=95)
        print(f"  saved {out_path}")

    print(f"\nDone. Grids in: {GRID_DIR}")


if __name__ == "__main__":
    main()