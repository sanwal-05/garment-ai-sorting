import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
IMG_SIZE = 224
tf.keras.utils.set_random_seed(SEED)

BASE_DIR = "/Users/sanjanabhowal/Desktop/DL-project/data"
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

label_maps = {
    "condition": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    "pilling": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    "stains": {"None": 0, "Minor": 1, "Major": 2},
    "holes": {"None": 0, "Minor": 1, "Major": 2}
}

backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
backbone.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
stem_out = backbone.get_layer("block6c_add").output
top_out = backbone.output
gap_out = tf.keras.layers.GlobalAveragePooling2D()(top_out)
gmp_out = tf.keras.layers.GlobalMaxPooling2D()(top_out)

extractor = tf.keras.Model(inputs=backbone.input, outputs=[stem_out, gap_out, gmp_out])

def load_img(path, aug_mode=0):
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    if aug_mode == 1:
        img = tf.image.flip_left_right(img)
        # Zoom/crop 92% center and resize back to 224x224
        img = tf.image.central_crop(img, 0.92)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def extract_for_paths(paths, aug_mode=0):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: load_img(p, aug_mode=aug_mode), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)
    stem_feat, gap_feat, gmp_feat = extractor.predict(ds, verbose=0)
    return stem_feat.astype(np.float16), gap_feat.astype(np.float32), gmp_feat.astype(np.float32)

for split in ["train", "val", "test"]:
    t0 = time.time()
    df = pd.read_csv(os.path.join(BASE_DIR, f"{split}.csv"), keep_default_na=False)
    for col, mapping in label_maps.items():
        df[col] = df[col].astype(int if col in ["condition", "pilling"] else str).map(mapping).astype(np.int32)

    f_stem, f_gap, f_gmp = extract_for_paths(df["front_image"].to_numpy(), aug_mode=0)
    b_stem, b_gap, b_gmp = extract_for_paths(df["back_image"].to_numpy(), aug_mode=0)

    save_dict = {
        "f_stem": f_stem, "f_gap": f_gap, "f_gmp": f_gmp,
        "b_stem": b_stem, "b_gap": b_gap, "b_gmp": b_gmp,
        "condition": df["condition"].to_numpy(dtype=np.int32),
        "pilling": df["pilling"].to_numpy(dtype=np.int32),
        "stains": df["stains"].to_numpy(dtype=np.int32),
        "holes": df["holes"].to_numpy(dtype=np.int32),
    }

    if split == "train":
        f_stem_aug, f_gap_aug, f_gmp_aug = extract_for_paths(df["front_image"].to_numpy(), aug_mode=1)
        b_stem_aug, b_gap_aug, b_gmp_aug = extract_for_paths(df["back_image"].to_numpy(), aug_mode=1)
        save_dict.update({
            "f_stem_aug": f_stem_aug, "f_gap_aug": f_gap_aug, "f_gmp_aug": f_gmp_aug,
            "b_stem_aug": b_stem_aug, "b_gap_aug": b_gap_aug, "b_gmp_aug": b_gmp_aug,
        })

    np.savez(os.path.join(CACHE_DIR, f"{split}_features.npz"), **save_dict)
    print(f"Extracted and saved {split} ({len(df)} records) in {time.time() - t0:.1f}s")
