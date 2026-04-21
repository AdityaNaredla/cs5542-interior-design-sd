"""
inspect_mat.py
Diagnostic — prints the actual structure of SUNRGBDMeta2DBB_v2.mat
so we can see the real field names and write a correct extractor.

Run: python inspect_mat.py
"""

from scipy.io import loadmat

MAT_PATH = "data/sun_rgbd_raw/SUNRGBDMeta2DBB_v2.mat"

print(f"Loading {MAT_PATH}...")
mat = loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)

print("\n" + "=" * 60)
print("TOP-LEVEL KEYS")
print("=" * 60)
for k in mat.keys():
    if not k.startswith("__"):
        val = mat[k]
        print(f"  {k}: type={type(val).__name__}, "
              f"shape={getattr(val, 'shape', 'N/A')}")

candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
if not candidate_keys:
    print("No data keys found!")
    exit(1)

main_key = candidate_keys[0]
print(f"\nUsing main key: '{main_key}'")
data = mat[main_key]

print("\n" + "=" * 60)
print("FIRST ENTRY: data[0]")
print("=" * 60)
first = data[0] if hasattr(data, "__len__") else data

print(f"Type: {type(first).__name__}")
print("Available attributes (fields):")
if hasattr(first, "_fieldnames"):
    for field in first._fieldnames:
        val = getattr(first, field, None)
        val_type = type(val).__name__
        val_preview = ""
        if val is not None:
            try:
                val_str = str(val)
                val_preview = val_str[:120] + ("..." if len(val_str) > 120 else "")
            except Exception:
                val_preview = "<could not preview>"
        print(f"  .{field}  (type={val_type})")
        print(f"      value: {val_preview}")
else:
    print("  No _fieldnames. dir() output:")
    print(" ", [a for a in dir(first) if not a.startswith("_")])

print("\n" + "=" * 60)
print("ENTRY 10 for comparison")
print("=" * 60)
try:
    tenth = data[10]
    if hasattr(tenth, "_fieldnames"):
        for field in tenth._fieldnames:
            val = getattr(tenth, field, None)
            val_str = str(val)[:100]
            print(f"  .{field}: {val_str}")
except Exception as e:
    print(f"Could not access entry 10: {e}")