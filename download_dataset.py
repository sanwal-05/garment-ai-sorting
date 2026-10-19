import os
import json
import time
import random
import struct
import zlib
import hashlib
import threading
import requests
import pandas as pd
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from remotezip import RemoteZip
from sklearn.model_selection import train_test_split

BASE_DIR = "/Users/sanjanabhowal/Desktop/DL-project/data"
OUTPUT_DIR = os.path.join(BASE_DIR, "garment_subset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_RECORDS = 5000
SEED = 42

print("1. Fetching Zenodo record URL...")
record = requests.get("https://zenodo.org/api/records/12518734", timeout=30).json()
file_info = next(f for f in record["files"] if f["key"] == "circular_fashion_v2.zip")
zip_url = file_info["links"]["self"]

print("2. Reading ZIP central directory via RemoteZip...")
t0 = time.time()
with RemoteZip(zip_url) as z:
    names = z.namelist()
    name_set = set(names)
    infolist = {info.filename: (info.header_offset, info.compress_size, info.compress_type) for info in z.infolist()}

print(f"Central directory loaded in {time.time() - t0:.1f}s ({len(names)} entries)")

candidate_paths = [
    p for p in names
    if p.startswith("station1/")
    and "/json_errors/" not in p
    and p.lower().endswith(".json")
    and "/labels_" in p
]

rng = random.Random(SEED)
sampled_paths = rng.sample(candidate_paths, min(MAX_RECORDS, len(candidate_paths)))
print(f"Available annotation files: {len(candidate_paths)} | Randomly selected: {len(sampled_paths)}")

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=5)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        thread_local.session = s
    return thread_local.session

def fetch_zip_entry_bytes(filename):
    header_offset, compress_size, compress_type = infolist[filename]
    # Local file header is 30 bytes + filename len + extra field len (usually < 256 bytes)
    end_byte = header_offset + 30 + len(filename.encode("utf-8")) + 256 + compress_size
    headers = {"Range": f"bytes={header_offset}-{end_byte}"}
    s = get_session()
    for attempt in range(5):
        try:
            resp = s.get(zip_url, headers=headers, timeout=30)
            if resp.status_code in (200, 206):
                buf = resp.content
                sig, _, _, _, _, _, _, _, _, fname_len, extra_len = struct.unpack("<4sHHHHHIIIHH", buf[:30])
                data_start = 30 + fname_len + extra_len
                comp_data = buf[data_start : data_start + compress_size]
                if len(comp_data) < compress_size:
                    # Rare case if extra_len > 256: fetch exact
                    end_byte2 = header_offset + data_start + compress_size
                    resp2 = s.get(zip_url, headers={"Range": f"bytes={header_offset + data_start}-{end_byte2 - 1}"}, timeout=30)
                    comp_data = resp2.content
                if compress_type == 0:
                    return comp_data
                elif compress_type == 8:
                    return zlib.decompress(comp_data, -15)
                else:
                    raise ValueError(f"Unsupported compression type: {compress_type}")
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {filename}")

def process_item(idx_and_path):
    idx, json_path = idx_and_path
    local_failed = []
    try:
        raw_json = fetch_zip_entry_bytes(json_path)
        meta = json.loads(raw_json.decode("utf-8"))
        if not isinstance(meta, dict):
            return idx, None, local_failed

        required = ["condition", "pilling", "stains", "holes"]
        if any(meta.get(k) is None or str(meta.get(k)).strip() == "" for k in required):
            return idx, None, local_failed

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
            # Check if already downloaded and valid
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                image_paths[view] = output_path
                continue
            try:
                img_bytes = fetch_zip_entry_bytes(image_path)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                img.save(output_path, quality=90)
                image_paths[view] = output_path
            except Exception as e:
                local_failed.append((image_path, str(e)[:150]))

        if not image_paths:
            return idx, None, local_failed

        record_dict = {
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
        }
        return idx, record_dict, local_failed
    except Exception as e:
        local_failed.append((json_path, str(e)[:150]))
        return idx, None, local_failed

print("3. Downloading 5,000 sampled records and images in parallel...")
t_dl = time.time()
results_by_idx = {}
all_failed = []

with ThreadPoolExecutor(max_workers=24) as executor:
    futures = {executor.submit(process_item, (i, p)): i for i, p in enumerate(sampled_paths)}
    done_count = 0
    for fut in as_completed(futures):
        idx, rec, fl = fut.result()
        results_by_idx[idx] = rec
        all_failed.extend(fl)
        done_count += 1
        if done_count % 250 == 0 or done_count == len(sampled_paths):
            elapsed = time.time() - t_dl
            usable_so_far = sum(1 for v in results_by_idx.values() if v is not None)
            print(f"  Completed {done_count}/{len(sampled_paths)} ({elapsed:.1f}s) | Usable so far: {usable_so_far} | Failed: {len(all_failed)}")

# Preserve exact order of sampled_paths!
records = [results_by_idx[i] for i in range(len(sampled_paths)) if results_by_idx[i] is not None]
df = pd.DataFrame(records)

csv_path = os.path.join(BASE_DIR, "garment_subset_metadata.csv")
df.to_csv(csv_path, index=False)

print("\n--- SUBSET SUMMARY ---")
print("Usable garments:", len(df))
print("Front images:", (df["front_image"] != "").sum() if len(df) else 0)
print("Back images:", (df["back_image"] != "").sum() if len(df) else 0)
print("Failed reads/images:", len(all_failed))

# Now run exact Cell 14 deduplication and split
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
