"""
smoke_test.py — Phase 1 verification
Confirms that PyTorch + CUDA + Diffusers are working end-to-end
by generating a single 512x512 image with Stable Diffusion v1.5.

Run: python smoke_test.py
Expected: `smoke_test_output.png` appears in the current directory.
First run will download ~4GB of model weights; subsequent runs reuse the cache.
"""

import torch
from diffusers import StableDiffusionPipeline

print("=" * 60)
print("CS 5542 — Phase 1 Smoke Test")
print("=" * 60)

# 1. CUDA check
assert torch.cuda.is_available(), "CUDA is not available. Check your PyTorch install."
device = "cuda"
print(f"✓ CUDA available")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  VRAM: {vram_gb:.1f} GB")

# 2. Load pipeline
print("\nLoading Stable Diffusion v1.5 (downloads ~4GB on first run)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,   # fp16 = half the VRAM, negligible quality loss
    safety_checker=None,          # skip for speed; re-enable for production
)
pipe = pipe.to(device)

# VRAM-saving tricks (harmless to enable even if you don't need them)
if vram_gb < 12:
    pipe.enable_attention_slicing()
    print("  Enabled attention slicing (VRAM < 12GB)")

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("  Enabled xformers memory-efficient attention")
except Exception:
    print("  xformers not active (OK — not required)")

# 3. Generate
prompt = "a cozy scandinavian bedroom with oak furniture and natural daylight, interior design photography, 8k, highly detailed"
negative_prompt = "blurry, distorted, low quality, warped furniture, text, watermark"

print(f"\nGenerating image...")
print(f"  Prompt: {prompt}")

generator = torch.Generator(device=device).manual_seed(42)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=generator,
).images[0]

# 4. Save
out_path = "smoke_test_output.png"
image.save(out_path)
print(f"\n✓ Success! Image saved to: {out_path}")
print("\nPhase 1 complete. You're ready for Phase 2.")