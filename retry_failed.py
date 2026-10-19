import os
import json
import time
import random
import hashlib
import requests
import pandas as pd
from io import BytesIO
from PIL import Image
from remotezip import RemoteZip
from sklearn.model_selection import train_test_split

BASE_DIR = "/Users/sanjanabhowal/Desktop/DL-project/data"
OUTPUT_DIR = os.path.join(BASE_DIR, "garment_subset")

record = requests.get("https://zenodo.org/api/records/12518734", timeout=30).json()
file_info = next(f for f in record["files"] if f["key"] == "circular_fashion_v2.zip")
zip_url = file_info["links"]["self"]

existing_df = pd.read_csv(os.path.join(BASE_DIR, "garment_subset_metadata.csv"), keep_default_na=False)
existing_by_path = {}
for _, row in existing_df.iterrows():
    if os.path.isfile(row["front_image"]) and os.path.isfile(row["back_image"]):
        existing_by_path[row["metadata_path"]] = row.to_dict()

print("Valid cached complete records:", len(existing_by_path))

with RemoteZip(zip_url) as z:
    names = z.namelist()
    name_set = set(names)
    candidate_paths = [
        p for p in names
        if p.startswith("station1/")
        and "/json_errors/" not in p
        and p.lower().endswith(".json")
        and "/labels_" in p
    ]
    rng = random.Random(42)
    sampled_paths = rng.sample(candidate_paths, min(5000, len(candidate_paths)))

    records = []
    failed = []
    retried = 0
    for i, json_path in enumerate(sampled_paths, start=1):
        if json_path in existing_by_path:
            records.append(existing_by_path[json_path])
            continue

        try:
            with z.open(json_path) as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                continue
            required = ["condition", "pilling", "stains", "holes"]
            if any(meta.get(k) is None or str(meta.get(k)).strip() == "" for k in required):
                continue

            folder = os.path.dirname(json_path)
            timestamp = os.path.basename(json_path).removeprefix("labels_").removesuffix(".json")
            garment_id = folder.replace("/", "_") + "_" + timestamp
            image_paths = {}

            for view in ["front", "back"]:
                image_path = f"{folder}/{view}_{timestamp}.jpg"
                if image_path not in name_set:
                    continue
                output_name = f"{garment_id}_{view}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    image_paths[view] = output_path
                    continue
                try:
                    retried += 1
                    with z.open(image_path) as image_file:
                        img = Image.open(BytesIO(image_file.read())).convert("RGB")
                    img.save(output_path, quality=90)
                    image_paths[view] = output_path
                except Exception as e:
                    failed.append((image_path, str(e)[:150]))

            if not image_paths:
                continue

            records.append({
                "garment_id": garment_id,
                "metadata_path": json_path,
                "condition": int(meta["condition"]),
                "pilling": int(meta["pilling"]),
                "stains": str(meta["stains"]).strip().title(),
                "holes": str(meta["holes"]).strip().title(),
                "usage": str(meta.get("usage", "")).strip(),
                "type": str(meta.get("type", "")).strip(),
                "front_image": image_paths.get("front", ""),
                "back_image": image_paths.get("back", "")
            })
        except Exception as e:
            failed.append((json_path, str(e)[:150]))

df = pd.DataFrame(records)
csv_path = os.path.join(BASE_DIR, "garment_subset_metadata.csv")
df.to_csv(csv_path, index=False)

print(f"Retried images: {retried} | Failed: {len(failed)}")
print("Usable garments:", len(df))
print("Front images:", (df["front_image"] != "").sum())
print("Back images:", (df["back_image"] != "").sum())

# Deduplicate and split
df = pd.read_csv(csv_path, keep_default_na=False)
df["condition"] = df["condition"].astype(int)
df["pilling"] = df["pilling"].astype(int)

df = df[
    df["front_image"].apply(os.path.isfile)
    & df["back_image"].apply(os.path.isfile)
].copy()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

seen_hashes = set()
keep_indices = []
duplicate_garments = []

for idx, row in df.iterrows():
    hashes = {
        sha256_file(row["front_image"]),
        sha256_file(row["back_image"])
    }
    if hashes & seen_hashes:
        duplicate_garments.append(row["garment_id"])
        continue
    keep_indices.append(idx)
    seen_hashes.update(hashes)

df_clean = df.loc[keep_indices].reset_index(drop=True)
print("\nOriginal garments:", len(df))
print("Duplicate garment records removed:", len(duplicate_garments))
print("Clean garments:", len(df_clean))
print("Removed IDs:", duplicate_garments)

train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.30,
    random_state=42,
    stratify=df_clean["condition"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["condition"]
)

train_df.to_csv(os.path.join(BASE_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(BASE_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(BASE_DIR, "test.csv"), index=False)

print("\nTraining:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

for split_name, split_df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
    print(f"\n{split_name} label distributions")
    for col in ["condition", "pilling", "stains", "holes"]:
        print(col, split_df[col].value_counts().to_dict())
