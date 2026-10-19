import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = "/Users/sanjanabhowal/Desktop/DL-project"
BASE_DIR = os.path.join(PROJECT_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

for d in [CHECKPOINT_DIR, FIG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

TASKS = ["condition", "pilling", "stains", "holes"]
NUM_CLASSES = {"condition": 5, "pilling": 5, "stains": 3, "holes": 3}
CLASS_NAMES = {
    "condition": ["1", "2", "3", "4", "5"],
    "pilling": ["1", "2", "3", "4", "5"],
    "stains": ["None", "Minor", "Major"],
    "holes": ["None", "Minor", "Major"]
}

LABEL_MAPS = {
    "condition": {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
    "pilling": {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
    "stains": {"None": 0, "Minor": 1, "Major": 2},
    "holes": {"None": 0, "Minor": 1, "Major": 2},
    "inverse": {
        "condition": {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},
        "pilling": {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},
        "stains": {0: "None", 1: "Minor", 2: "Major"},
        "holes": {0: "None", 1: "Minor", 2: "Major"}
    }
}

with open(os.path.join(RESULTS_DIR, "label_mappings.json"), "w") as f:
    json.dump(LABEL_MAPS, f, indent=2)

print("Loading cached features...")
train_data = np.load(os.path.join(CACHE_DIR, "train_features.npz"))
val_data = np.load(os.path.join(CACHE_DIR, "val_features.npz"))
test_data = np.load(os.path.join(CACHE_DIR, "test_features.npz"))

y_train = {t: train_data[t].astype(np.int32) for t in TASKS}
y_val = {t: val_data[t].astype(np.int32) for t in TASKS}
y_test = {t: test_data[t].astype(np.int32) for t in TASKS}

def get_sample_weights(y_dict, power=1.0, max_clip=None):
    sw = {}
    for t in TASKS:
        classes = np.unique(y_dict[t])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_dict[t])
        weights = np.power(weights, power)
        if max_clip is not None:
            weights = np.minimum(weights, max_clip)
        weights = weights / np.mean(weights[y_dict[t]])  # normalize mean weight to 1.0
        w_map = dict(zip(classes, weights))
        sw[t] = np.array([w_map[y] for y in y_dict[t]], dtype=np.float32)
    return sw

def get_raw_balanced_weights(y_dict):
    sw = {}
    for t in TASKS:
        classes = np.unique(y_dict[t])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_dict[t])
        w_map = dict(zip(classes, weights))
        sw[t] = np.array([w_map[y] for y in y_dict[t]], dtype=np.float32)
    return sw

def evaluate_predictions(y_true_dict, preds_dict):
    metrics = {}
    macro_f1s = []
    for t in TASKS:
        yt = y_true_dict[t]
        yp = np.argmax(preds_dict[t], axis=1) if preds_dict[t].ndim > 1 else preds_dict[t]
        acc = accuracy_score(yt, yp)
        mf1 = f1_score(yt, yp, average="macro", zero_division=0)
        wf1 = f1_score(yt, yp, average="weighted", zero_division=0)
        prec, rec, f1_c, sup = precision_recall_fscore_support(
            yt, yp, labels=list(range(NUM_CLASSES[t])), zero_division=0
        )
        cm = confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES[t])))
        metrics[t] = {
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "weighted_f1": float(wf1),
            "per_class": {
                CLASS_NAMES[t][i]: {
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1_c[i]),
                    "support": int(sup[i])
                }
                for i in range(NUM_CLASSES[t])
            },
            "confusion_matrix": cm.tolist()
        }
        macro_f1s.append(mf1)
    metrics["mean_macro_f1"] = float(np.mean(macro_f1s))
    return metrics

class ValidationMacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_x, val_y, patience=5):
        super().__init__()
        self.val_x = val_x
        self.val_y = val_y
        self.patience = patience
        self.best_mean_f1 = -1.0
        self.best_epoch = 0
        self.best_weights = None
        self.best_task_f1s = {}
        self.history_f1 = {t: [] for t in TASKS}
        self.history_f1["mean_macro_f1"] = []
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_x, batch_size=64, verbose=0)
        task_f1s = {}
        for t in TASKS:
            yp = np.argmax(preds[t], axis=1)
            mf1 = f1_score(self.val_y[t], yp, average="macro", zero_division=0)
            task_f1s[t] = float(mf1)
            self.history_f1[t].append(float(mf1))
        mean_f1 = float(np.mean(list(task_f1s.values())))
        self.history_f1["mean_macro_f1"].append(mean_f1)
        if logs is not None:
            logs["val_mean_macro_f1"] = mean_f1
            for t in TASKS:
                logs[f"val_{t}_macro_f1"] = task_f1s[t]

        if mean_f1 > self.best_mean_f1:
            self.best_mean_f1 = mean_f1
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.best_task_f1s = task_f1s
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

# Pre-load EfficientNetB0 once to extract top blocks when needed
base_effnet = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)

def build_cloned_top_block(trainable=True):
    # Clone layers from block6d_expand_conv (index 207) to top_activation (index 237)
    # Using a functional model that shares weights cloned from base_effnet
    backbone_fresh = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    sub = Model(
        inputs=backbone_fresh.get_layer("block6c_add").output,
        outputs=backbone_fresh.output,
        name="efficientnet_top_blocks"
    )
    for layer in sub.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = trainable
    return sub

experiment_records = []
detailed_results = {}

# =========================================================================
# E0: Majority Class Baseline
# =========================================================================
print("\n--- Running E0: Majority Baseline ---")
val_maj_preds = {}
test_maj_preds = {}
for t in TASKS:
    maj_c = int(np.bincount(y_train[t]).argmax())
    val_maj_preds[t] = np.full_like(y_val[t], maj_c)
    test_maj_preds[t] = np.full_like(y_test[t], maj_c)

e0_val_metrics = evaluate_predictions(y_val, val_maj_preds)
e0_test_metrics = evaluate_predictions(y_test, test_maj_preds)

experiment_records.append({
    "exp_id": "E0_Majority",
    "description": "Trivial majority-class predictor (predicts most frequent training class)",
    "changes_from_baseline": "No neural network; predicts mode class per task",
    "epochs_trained": 0,
    "best_epoch": 0,
    "train_time_sec": 0.0,
    "val_mean_macro_f1": e0_val_metrics["mean_macro_f1"],
    "val_condition_macro_f1": e0_val_metrics["condition"]["macro_f1"],
    "val_pilling_macro_f1": e0_val_metrics["pilling"]["macro_f1"],
    "val_stains_macro_f1": e0_val_metrics["stains"]["macro_f1"],
    "val_holes_macro_f1": e0_val_metrics["holes"]["macro_f1"],
    "val_preds": val_maj_preds,
    "test_preds": test_maj_preds
})

# =========================================================================
# E1: Exact Reproduction of Original Baseline Architecture & Training Setup
# =========================================================================
print("\n--- Running E1: Original Baseline Reproduction ---")
tf.keras.utils.set_random_seed(SEED)
t0 = time.time()
f_gap_in = layers.Input(shape=(1280,), name="f_gap")
b_gap_in = layers.Input(shape=(1280,), name="b_gap")
comb = layers.Concatenate()([f_gap_in, b_gap_in])
comb = layers.Dropout(0.3)(comb)
comb = layers.Dense(256, activation="relu")(comb)
comb = layers.Dropout(0.2)(comb)
outs = {
    t: layers.Dense(NUM_CLASSES[t], activation="softmax", name=t)(comb)
    for t in TASKS
}
m_e1 = Model([f_gap_in, b_gap_in], outs, name="E1_Baseline")
m_e1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={t: "sparse_categorical_crossentropy" for t in TASKS},
    metrics={t: ["accuracy"] for t in TASKS}
)

sw_raw = get_raw_balanced_weights(y_train)
ds_e1_train = tf.data.Dataset.from_tensor_slices((
    {"f_gap": train_data["f_gap"], "b_gap": train_data["b_gap"]},
    y_train,
    sw_raw
)).shuffle(len(y_train["condition"]), seed=SEED).batch(8)

ds_e1_val = tf.data.Dataset.from_tensor_slices((
    {"f_gap": val_data["f_gap"], "b_gap": val_data["b_gap"]},
    y_val
)).batch(64)

cb_f1_e1 = ValidationMacroF1Callback(
    {"f_gap": val_data["f_gap"], "b_gap": val_data["b_gap"]}, y_val, patience=10
)
cb_es_e1 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

hist_e1 = m_e1.fit(ds_e1_train, validation_data=ds_e1_val, epochs=10, callbacks=[cb_f1_e1, cb_es_e1], verbose=0)
t_e1 = time.time() - t0
# Note: in E1 baseline, EarlyStopping restores best val_loss weights!
# Let's re-apply cb_es_e1 best weights to evaluate faithful baseline behavior
if cb_es_e1.best_weights is not None:
    m_e1.set_weights(cb_es_e1.best_weights)

val_preds_e1 = m_e1.predict({"f_gap": val_data["f_gap"], "b_gap": val_data["b_gap"]}, batch_size=64, verbose=0)
test_preds_e1 = m_e1.predict({"f_gap": test_data["f_gap"], "b_gap": test_data["b_gap"]}, batch_size=64, verbose=0)
e1_val_m = evaluate_predictions(y_val, val_preds_e1)

print(f"E1 finished in {t_e1:.1f}s | Val Mean Macro-F1: {e1_val_m['mean_macro_f1']:.4f}")
experiment_records.append({
    "exp_id": "E1_Baseline_Repro",
    "description": "Original baseline (frozen EfficientNet-B0 GAP, raw balanced weights, batch=8, lr=1e-3, monitor=val_loss)",
    "changes_from_baseline": "Exact reproduction on local cached split",
    "epochs_trained": len(hist_e1.history["loss"]),
    "best_epoch": int(np.argmin(hist_e1.history["val_loss"]) + 1),
    "train_time_sec": round(t_e1, 2),
    "val_mean_macro_f1": e1_val_m["mean_macro_f1"],
    "val_condition_macro_f1": e1_val_m["condition"]["macro_f1"],
    "val_pilling_macro_f1": e1_val_m["pilling"]["macro_f1"],
    "val_stains_macro_f1": e1_val_m["stains"]["macro_f1"],
    "val_holes_macro_f1": e1_val_m["holes"]["macro_f1"],
    "val_preds": val_preds_e1,
    "test_preds": test_preds_e1,
    "history_f1": cb_f1_e1.history_f1
})

# =========================================================================
# E2: Tempered Class Weights + Val Macro-F1 Selection + Cosine LR + BatchNorm
# =========================================================================
print("\n--- Running E2: Tempered Class Weights + Macro-F1 Selection + BN Head ---")
tf.keras.utils.set_random_seed(SEED)
t0 = time.time()
f_gap_in = layers.Input(shape=(1280,), name="f_gap")
b_gap_in = layers.Input(shape=(1280,), name="b_gap")
comb = layers.Concatenate()([f_gap_in, b_gap_in])
comb = layers.BatchNormalization()(comb)
comb = layers.Dropout(0.35)(comb)
comb = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(comb)
comb = layers.BatchNormalization()(comb)
comb = layers.Dropout(0.25)(comb)
outs = {
    t: layers.Dense(NUM_CLASSES[t], activation="softmax", name=t)(comb)
    for t in TASKS
}
m_e2 = Model([f_gap_in, b_gap_in], outs, name="E2_TemperedWeights")

steps_per_epoch = int(np.ceil(len(y_train["condition"]) / 32))
lr_sched_e2 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=steps_per_epoch * 20, alpha=0.05
)
m_e2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched_e2),
    loss={t: "sparse_categorical_crossentropy" for t in TASKS},
    loss_weights={"condition": 1.2, "pilling": 1.2, "stains": 1.0, "holes": 1.0},
    metrics={t: ["accuracy"] for t in TASKS}
)

# Tempered weights (power=0.55) prevents minority class over-prediction while boosting recall
sw_temp = get_sample_weights(y_train, power=0.55, max_clip=4.5)
ds_e2_train = tf.data.Dataset.from_tensor_slices((
    {"f_gap": train_data["f_gap"], "b_gap": train_data["b_gap"]},
    y_train,
    sw_temp
)).shuffle(len(y_train["condition"]), seed=SEED).batch(32)

cb_f1_e2 = ValidationMacroF1Callback(
    {"f_gap": val_data["f_gap"], "b_gap": val_data["b_gap"]}, y_val, patience=6
)
hist_e2 = m_e2.fit(ds_e2_train, epochs=20, callbacks=[cb_f1_e2], verbose=0)
t_e2 = time.time() - t0

val_preds_e2 = m_e2.predict({"f_gap": val_data["f_gap"], "b_gap": val_data["b_gap"]}, batch_size=64, verbose=0)
test_preds_e2 = m_e2.predict({"f_gap": test_data["f_gap"], "b_gap": test_data["b_gap"]}, batch_size=64, verbose=0)
e2_val_m = evaluate_predictions(y_val, val_preds_e2)

print(f"E2 finished in {t_e2:.1f}s (best ep {cb_f1_e2.best_epoch}) | Val Mean Macro-F1: {e2_val_m['mean_macro_f1']:.4f}")
experiment_records.append({
    "exp_id": "E2_Tempered_MacroF1",
    "description": "Tempered class weights (power=0.55), BatchNorm+L2 head, batch=32, CosineDecay LR, checkpointing on val Macro-F1",
    "changes_from_baseline": "+Tempered weights (p=0.55), +Val Macro-F1 early stopping, +BatchNorm/L2, +Cosine LR, +batch=32",
    "epochs_trained": len(hist_e2.history["loss"]),
    "best_epoch": cb_f1_e2.best_epoch,
    "train_time_sec": round(t_e2, 2),
    "val_mean_macro_f1": e2_val_m["mean_macro_f1"],
    "val_condition_macro_f1": e2_val_m["condition"]["macro_f1"],
    "val_pilling_macro_f1": e2_val_m["pilling"]["macro_f1"],
    "val_stains_macro_f1": e2_val_m["stains"]["macro_f1"],
    "val_holes_macro_f1": e2_val_m["holes"]["macro_f1"],
    "val_preds": val_preds_e2,
    "test_preds": test_preds_e2,
    "history_f1": cb_f1_e2.history_f1
})

# =========================================================================
# E3: Multi-Scale Spatial Pooling (GAP + GMP + |Front-Back| Difference) + Task-Specific Branches
# =========================================================================
print("\n--- Running E3: Multi-Scale Pooling (GAP+GMP+Diff) + Task-Specific Towers ---")
tf.keras.utils.set_random_seed(SEED)
t0 = time.time()

f_gap_in = layers.Input(shape=(1280,), name="f_gap")
b_gap_in = layers.Input(shape=(1280,), name="b_gap")
f_gmp_in = layers.Input(shape=(1280,), name="f_gmp")
b_gmp_in = layers.Input(shape=(1280,), name="b_gmp")

diff_gap = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="abs_diff_gap")([f_gap_in, b_gap_in])
diff_gmp = layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="abs_diff_gmp")([f_gmp_in, b_gmp_in])

# Global representation (GAP + diff_gap) and localized defect representation (GMP + diff_gmp)
shared_trunk = layers.Concatenate()([f_gap_in, b_gap_in, f_gmp_in, b_gmp_in, diff_gap, diff_gmp])
shared_trunk = layers.BatchNormalization()(shared_trunk)
shared_trunk = layers.Dropout(0.4)(shared_trunk)
shared_trunk = layers.Dense(384, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(shared_trunk)
shared_trunk = layers.BatchNormalization()(shared_trunk)
shared_trunk = layers.Dropout(0.3)(shared_trunk)

outs_e3 = {}
for t in TASKS:
    branch = layers.Dense(128, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(shared_trunk)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Dropout(0.2)(branch)
    outs_e3[t] = layers.Dense(NUM_CLASSES[t], activation="softmax", name=t)(branch)

m_e3 = Model([f_gap_in, b_gap_in, f_gmp_in, b_gmp_in], outs_e3, name="E3_MultiScale_Towers")

lr_sched_e3 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=steps_per_epoch * 22, alpha=0.02
)
m_e3.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_sched_e3, weight_decay=1e-4),
    loss={t: "sparse_categorical_crossentropy" for t in TASKS},
    loss_weights={"condition": 1.25, "pilling": 1.25, "stains": 1.0, "holes": 1.0},
    metrics={t: ["accuracy"] for t in TASKS}
)

x_train_e3 = {
    "f_gap": train_data["f_gap"], "b_gap": train_data["b_gap"],
    "f_gmp": train_data["f_gmp"], "b_gmp": train_data["b_gmp"]
}
x_val_e3 = {
    "f_gap": val_data["f_gap"], "b_gap": val_data["b_gap"],
    "f_gmp": val_data["f_gmp"], "b_gmp": val_data["b_gmp"]
}
x_test_e3 = {
    "f_gap": test_data["f_gap"], "b_gap": test_data["b_gap"],
    "f_gmp": test_data["f_gmp"], "b_gmp": test_data["b_gmp"]
}

ds_e3_train = tf.data.Dataset.from_tensor_slices((x_train_e3, y_train, sw_temp)).shuffle(len(y_train["condition"]), seed=SEED).batch(32)
cb_f1_e3 = ValidationMacroF1Callback(x_val_e3, y_val, patience=6)
hist_e3 = m_e3.fit(ds_e3_train, epochs=22, callbacks=[cb_f1_e3], verbose=0)
t_e3 = time.time() - t0

val_preds_e3 = m_e3.predict(x_val_e3, batch_size=64, verbose=0)
test_preds_e3 = m_e3.predict(x_test_e3, batch_size=64, verbose=0)
e3_val_m = evaluate_predictions(y_val, val_preds_e3)

print(f"E3 finished in {t_e3:.1f}s (best ep {cb_f1_e3.best_epoch}) | Val Mean Macro-F1: {e3_val_m['mean_macro_f1']:.4f}")
experiment_records.append({
    "exp_id": "E3_GAP_GMP_Towers",
    "description": "GAP + GlobalMaxPooling2D (localized defects) + |Front-Back| diff + task-specific Swish towers + AdamW",
    "changes_from_baseline": "E2 + GlobalMaxPooling2D + |Front-Back| feature diff + task-specific 128-d towers + AdamW",
    "epochs_trained": len(hist_e3.history["loss"]),
    "best_epoch": cb_f1_e3.best_epoch,
    "train_time_sec": round(t_e3, 2),
    "val_mean_macro_f1": e3_val_m["mean_macro_f1"],
    "val_condition_macro_f1": e3_val_m["condition"]["macro_f1"],
    "val_pilling_macro_f1": e3_val_m["pilling"]["macro_f1"],
    "val_stains_macro_f1": e3_val_m["stains"]["macro_f1"],
    "val_holes_macro_f1": e3_val_m["holes"]["macro_f1"],
    "val_preds": val_preds_e3,
    "test_preds": test_preds_e3,
    "history_f1": cb_f1_e3.history_f1
})

# =========================================================================
# E4: Moderate Augmentation + Two-Stage Fine-Tuning of EfficientNet-B0 Top Blocks
#     with Separate Backbone / Head Learning Rates
# =========================================================================
print("\n--- Running E4: Augmentation + Fine-Tuning Top EfficientNetB0 Blocks (Separate LRs) ---")
tf.keras.utils.set_random_seed(SEED)
t0 = time.time()

# Combine original + augmented training views (2x training set with horizontal flip + 92% center crop)
f_stem_tr = np.concatenate([train_data["f_stem"].astype(np.float32), train_data["f_stem_aug"].astype(np.float32)], axis=0)
b_stem_tr = np.concatenate([train_data["b_stem"].astype(np.float32), train_data["b_stem_aug"].astype(np.float32)], axis=0)
y_train_aug = {t: np.concatenate([y_train[t], y_train[t]], axis=0) for t in TASKS}
sw_temp_aug = get_sample_weights(y_train_aug, power=0.52, max_clip=4.0)

f_stem_val = val_data["f_stem"].astype(np.float32)
b_stem_val = val_data["b_stem"].astype(np.float32)
f_stem_test = test_data["f_stem"].astype(np.float32)
b_stem_test = test_data["b_stem"].astype(np.float32)

top_block = build_cloned_top_block(trainable=False)

f_stem_in = layers.Input(shape=(7, 7, 192), name="f_stem")
b_stem_in = layers.Input(shape=(7, 7, 192), name="b_stem")

# Spatial dropout / Gaussian noise during training for regularization
f_s = layers.SpatialDropout2D(0.1)(f_stem_in)
b_s = layers.SpatialDropout2D(0.1)(b_stem_in)

f_top = top_block(f_s)
b_top = top_block(b_s)

f_gap = layers.GlobalAveragePooling2D()(f_top)
b_gap = layers.GlobalAveragePooling2D()(b_top)
f_gmp = layers.GlobalMaxPooling2D()(f_top)
b_gmp = layers.GlobalMaxPooling2D()(b_top)

diff_gap = layers.Subtract()([f_gap, b_gap])
diff_gap = layers.Lambda(lambda z: tf.abs(z))(diff_gap)

merged = layers.Concatenate()([f_gap, b_gap, f_gmp, b_gmp, diff_gap])
merged = layers.BatchNormalization(name="head_bn1")(merged)
merged = layers.Dropout(0.4, name="head_drop1")(merged)
merged = layers.Dense(384, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="head_dense1")(merged)
merged = layers.BatchNormalization(name="head_bn2")(merged)
merged = layers.Dropout(0.3, name="head_drop2")(merged)

outs_e4 = {}
for t in TASKS:
    br = layers.Dense(128, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f"head_{t}_fc")(merged)
    br = layers.BatchNormalization(name=f"head_{t}_bn")(br)
    br = layers.Dropout(0.2, name=f"head_{t}_drop")(br)
    outs_e4[t] = layers.Dense(NUM_CLASSES[t], activation="softmax", name=t)(br)

m_e4 = Model([f_stem_in, b_stem_in], outs_e4, name="E4_FineTuned_MultiTask")

# Stage 1: Warm up classification heads for 4 epochs (backbone top blocks frozen, head LR = 1e-3)
m_e4.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss={t: "sparse_categorical_crossentropy" for t in TASKS},
    loss_weights={"condition": 1.25, "pilling": 1.25, "stains": 1.0, "holes": 1.0},
    metrics={t: ["accuracy"] for t in TASKS}
)

ds_e4_train = tf.data.Dataset.from_tensor_slices((
    {"f_stem": f_stem_tr, "b_stem": b_stem_tr},
    y_train_aug,
    sw_temp_aug
)).shuffle(len(y_train_aug["condition"]), seed=SEED).batch(64)

cb_f1_e4_s1 = ValidationMacroF1Callback(
    {"f_stem": f_stem_val, "b_stem": b_stem_val}, y_val, patience=4
)
print("  E4 Stage 1: Head warmup (4 epochs, head_lr=1e-3, backbone frozen)...")
hist_e4_s1 = m_e4.fit(ds_e4_train, epochs=4, callbacks=[cb_f1_e4_s1], verbose=0)

# Stage 2: Unfreeze top blocks (block6d, block7a, top_conv; keep BatchNorm frozen)
# Implement differential learning rates via gradient scaling on backbone weights (0.2x head LR)
for layer in top_block.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

class DifferentialLRAdamW(tf.keras.optimizers.AdamW):
    def __init__(self, head_lr, backbone_lr_mult=0.2, **kwargs):
        super().__init__(learning_rate=head_lr, **kwargs)
        self.backbone_lr_mult = backbone_lr_mult

    def apply_gradients(self, grads_and_vars, **kwargs):
        scaled_gv = []
        for g, v in grads_and_vars:
            if g is not None and ("block6" in v.path or "block7" in v.path or "top_conv" in v.path):
                scaled_gv.append((g * self.backbone_lr_mult, v))
            else:
                scaled_gv.append((g, v))
        return super().apply_gradients(scaled_gv, **kwargs)

steps_s2 = int(np.ceil(len(y_train_aug["condition"]) / 64))
lr_sched_s2 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=4e-4, decay_steps=steps_s2 * 10, alpha=0.05
)
opt_diff = DifferentialLRAdamW(head_lr=lr_sched_s2, backbone_lr_mult=0.2, weight_decay=1e-4)

m_e4.compile(
    optimizer=opt_diff,
    loss={t: "sparse_categorical_crossentropy" for t in TASKS},
    loss_weights={"condition": 1.25, "pilling": 1.25, "stains": 1.0, "holes": 1.0},
    metrics={t: ["accuracy"] for t in TASKS}
)

cb_f1_e4_s2 = ValidationMacroF1Callback(
    {"f_stem": f_stem_val, "b_stem": b_stem_val}, y_val, patience=4
)
# Seed stage 2 callback with stage 1 best so it never degrades
cb_f1_e4_s2.best_mean_f1 = cb_f1_e4_s1.best_mean_f1
cb_f1_e4_s2.best_weights = m_e4.get_weights()
cb_f1_e4_s2.best_epoch = cb_f1_e4_s1.best_epoch

print("  E4 Stage 2: Fine-tuning block6d+block7a+top_conv (up to 8 epochs, head_lr=4e-4, backbone_lr=8e-5)...")
hist_e4_s2 = m_e4.fit(ds_e4_train, epochs=8, callbacks=[cb_f1_e4_s2], verbose=0)
t_e4 = time.time() - t0

val_preds_e4 = m_e4.predict({"f_stem": f_stem_val, "b_stem": b_stem_val}, batch_size=64, verbose=0)
test_preds_e4 = m_e4.predict({"f_stem": f_stem_test, "b_stem": b_stem_test}, batch_size=64, verbose=0)
e4_val_m = evaluate_predictions(y_val, val_preds_e4)

combined_hist_e4 = {
    k: cb_f1_e4_s1.history_f1[k] + cb_f1_e4_s2.history_f1[k]
    for k in cb_f1_e4_s1.history_f1
}
best_ep_e4 = int(np.argmax(combined_hist_e4["mean_macro_f1"]) + 1)

print(f"E4 finished in {t_e4:.1f}s (best ep {best_ep_e4}) | Val Mean Macro-F1: {e4_val_m['mean_macro_f1']:.4f}")
experiment_records.append({
    "exp_id": "E4_Aug_FineTune_DiffLR",
    "description": "Moderate augmentation (flip+92% crop) + 2-stage fine-tuning of block6d/block7a/top_conv with differential LR (head 4e-4, backbone 8e-5)",
    "changes_from_baseline": "E3 + Moderate augmentation + Unfreeze block6d/7a/top_conv + Differential LR (0.2x backbone)",
    "epochs_trained": len(combined_hist_e4["mean_macro_f1"]),
    "best_epoch": best_ep_e4,
    "train_time_sec": round(t_e4, 2),
    "val_mean_macro_f1": e4_val_m["mean_macro_f1"],
    "val_condition_macro_f1": e4_val_m["condition"]["macro_f1"],
    "val_pilling_macro_f1": e4_val_m["pilling"]["macro_f1"],
    "val_stains_macro_f1": e4_val_m["stains"]["macro_f1"],
    "val_holes_macro_f1": e4_val_m["holes"]["macro_f1"],
    "val_preds": val_preds_e4,
    "test_preds": test_preds_e4,
    "history_f1": combined_hist_e4
})

# =========================================================================
# E5: Single-View (Front Only) vs Two-View Ablation
# =========================================================================
print("\n--- Running E5: Single-View (Front Image Only) Ablation ---")
tf.keras.utils.set_random_seed(SEED)
t0 = time.time()

f_gap_in = layers.Input(shape=(1280,), name="f_gap")
f_gmp_in = layers.Input(shape=(1280,), name="f_gmp")
trunk_sv = layers.Concatenate()([f_gap_in, f_gmp_in])
trunk_sv = layers.BatchNormalization()(trunk_sv)
trunk_sv = layers.Dropout(0.4)(trunk_sv)
trunk_sv = layers.Dense(384, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(trunk_sv)
trunk_sv = layers.BatchNormalization()(trunk_sv)
trunk_sv = layers.Dropout(0.3)(trunk_sv)

outs_e5 = {}
for t in TASKS:
    br = layers.Dense(128, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(trunk_sv)
    br = layers.BatchNormalization()(br)
    br = layers.Dropout(0.2)(br)
    outs_e5[t] = layers.Dense(NUM_CLASSES[t], activation="softmax", name=t)(br)

m_e5 = Model([f_gap_in, f_gmp_in], outs_e5, name="E5_SingleView_FrontOnly")
lr_sched_e5 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=steps_per_epoch * 20, alpha=0.02
)
m_e5.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_sched_e5, weight_decay=1e-4),
    loss={t: "sparse_categorical_crossentropy" for t in TASKS},
    loss_weights={"condition": 1.25, "pilling": 1.25, "stains": 1.0, "holes": 1.0},
    metrics={t: ["accuracy"] for t in TASKS}
)

x_train_e5 = {"f_gap": train_data["f_gap"], "f_gmp": train_data["f_gmp"]}
x_val_e5 = {"f_gap": val_data["f_gap"], "f_gmp": val_data["f_gmp"]}
x_test_e5 = {"f_gap": test_data["f_gap"], "f_gmp": test_data["f_gmp"]}

ds_e5_train = tf.data.Dataset.from_tensor_slices((x_train_e5, y_train, sw_temp)).shuffle(len(y_train["condition"]), seed=SEED).batch(32)
cb_f1_e5 = ValidationMacroF1Callback(x_val_e5, y_val, patience=6)
hist_e5 = m_e5.fit(ds_e5_train, epochs=20, callbacks=[cb_f1_e5], verbose=0)
t_e5 = time.time() - t0

val_preds_e5 = m_e5.predict(x_val_e5, batch_size=64, verbose=0)
test_preds_e5 = m_e5.predict(x_test_e5, batch_size=64, verbose=0)
e5_val_m = evaluate_predictions(y_val, val_preds_e5)

print(f"E5 finished in {t_e5:.1f}s (best ep {cb_f1_e5.best_epoch}) | Val Mean Macro-F1: {e5_val_m['mean_macro_f1']:.4f}")
experiment_records.append({
    "exp_id": "E5_SingleView_Ablation",
    "description": "Single-view ablation (Front image only, same architecture/loss as E3)",
    "changes_from_baseline": "E3 without back_image (tests contribution of two-view fusion)",
    "epochs_trained": len(hist_e5.history["loss"]),
    "best_epoch": cb_f1_e5.best_epoch,
    "train_time_sec": round(t_e5, 2),
    "val_mean_macro_f1": e5_val_m["mean_macro_f1"],
    "val_condition_macro_f1": e5_val_m["condition"]["macro_f1"],
    "val_pilling_macro_f1": e5_val_m["pilling"]["macro_f1"],
    "val_stains_macro_f1": e5_val_m["stains"]["macro_f1"],
    "val_holes_macro_f1": e5_val_m["holes"]["macro_f1"],
    "val_preds": val_preds_e5,
    "test_preds": test_preds_e5,
    "history_f1": cb_f1_e5.history_f1
})

# =========================================================================
# E6: Validation Prior-Adjustment / Threshold Tuning on Best Fine-Tuned Model (E4)
#     (Using ONLY validation set to tune prior-scaling exponents tau per task)
# =========================================================================
print("\n--- Running E6: Validation-Tuned Prior Calibration on E4 ---")
t0 = time.time()
best_base_exp = max([experiment_records[2], experiment_records[3], experiment_records[4]], key=lambda r: r["val_mean_macro_f1"])
print(f"Base model for E6 calibration: {best_base_exp['exp_id']} (Val Mean Macro-F1 = {best_base_exp['val_mean_macro_f1']:.4f})")

val_probs_base = best_base_exp["val_preds"]
test_probs_base = best_base_exp["test_preds"]

val_preds_e6 = {}
test_preds_e6 = {}
best_gammas = {}

for t in TASKS:
    priors = np.bincount(y_train[t]) / len(y_train[t])
    best_g = 0.0
    best_mf1 = f1_score(y_val[t], np.argmax(val_probs_base[t], axis=1), average="macro", zero_division=0)
    # Search exponent gamma in [-0.6, 0.6] on validation set ONLY
    for g in np.linspace(-0.6, 0.6, 61):
        scale = np.power(priors + 1e-6, -g)
        adj_val = val_probs_base[t] * scale[None, :]
        mf1 = f1_score(y_val[t], np.argmax(adj_val, axis=1), average="macro", zero_division=0)
        if mf1 > best_mf1:
            best_mf1 = mf1
            best_g = float(g)
    best_gammas[t] = best_g
    scale = np.power(priors + 1e-6, -best_g)
    val_preds_e6[t] = (val_probs_base[t] * scale[None, :]) / np.sum(val_probs_base[t] * scale[None, :], axis=1, keepdims=True)
    test_preds_e6[t] = (test_probs_base[t] * scale[None, :]) / np.sum(test_probs_base[t] * scale[None, :], axis=1, keepdims=True)

t_e6 = time.time() - t0
e6_val_m = evaluate_predictions(y_val, val_preds_e6)
print(f"E6 finished in {t_e6:.2f}s | Tuned gammas: {best_gammas} | Val Mean Macro-F1: {e6_val_m['mean_macro_f1']:.4f}")

experiment_records.append({
    "exp_id": "E6_Full_Calibrated",
    "description": f"{best_base_exp['exp_id']} + Validation-tuned post-hoc prior calibration for rare defect classes",
    "changes_from_baseline": f"{best_base_exp['exp_id']} + per-task prior scaling tuned strictly on validation Macro-F1",
    "epochs_trained": best_base_exp["epochs_trained"],
    "best_epoch": best_base_exp["best_epoch"],
    "train_time_sec": round(best_base_exp["train_time_sec"] + t_e6, 2),
    "val_mean_macro_f1": e6_val_m["mean_macro_f1"],
    "val_condition_macro_f1": e6_val_m["condition"]["macro_f1"],
    "val_pilling_macro_f1": e6_val_m["pilling"]["macro_f1"],
    "val_stains_macro_f1": e6_val_m["stains"]["macro_f1"],
    "val_holes_macro_f1": e6_val_m["holes"]["macro_f1"],
    "val_preds": val_preds_e6,
    "test_preds": test_preds_e6,
    "history_f1": best_base_exp.get("history_f1", {}),
    "calibration_gammas": best_gammas
})

# =========================================================================
# Save Full End-to-End Best Model Checkpoint (.keras)
# =========================================================================
print("\nBuilding and saving full end-to-end best model checkpoint (224x224 image inputs -> 4 heads)...")
front_img_in = layers.Input(shape=(224, 224, 3), name="front_image")
back_img_in = layers.Input(shape=(224, 224, 3), name="back_image")

# Rebuild full EfficientNet-B0 with the trained top_block weights from E4
full_effnet = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
for layer in top_block.layers:
    try:
        target_layer = full_effnet.get_layer(layer.name)
        target_layer.set_weights(layer.get_weights())
    except Exception:
        pass

f_top_full = full_effnet(front_img_in, training=False)
b_top_full = full_effnet(back_img_in, training=False)

f_gap_full = layers.GlobalAveragePooling2D()(f_top_full)
b_gap_full = layers.GlobalAveragePooling2D()(b_top_full)
f_gmp_full = layers.GlobalMaxPooling2D()(f_top_full)
b_gmp_full = layers.GlobalMaxPooling2D()(b_top_full)
diff_gap_full = layers.Lambda(lambda z: tf.abs(z[0] - z[1]))([f_gap_full, b_gap_full])

m_full_x = layers.Concatenate()([f_gap_full, b_gap_full, f_gmp_full, b_gmp_full, diff_gap_full])
m_full_x = m_e4.get_layer("head_bn1")(m_full_x)
m_full_x = m_e4.get_layer("head_drop1")(m_full_x)
m_full_x = m_e4.get_layer("head_dense1")(m_full_x)
m_full_x = m_e4.get_layer("head_bn2")(m_full_x)
m_full_x = m_e4.get_layer("head_drop2")(m_full_x)

full_outs = {}
for t in TASKS:
    br = m_e4.get_layer(f"head_{t}_fc")(m_full_x)
    br = m_e4.get_layer(f"head_{t}_bn")(br)
    br = m_e4.get_layer(f"head_{t}_drop")(br)
    full_outs[t] = m_e4.get_layer(t)(br)

full_end_to_end_model = Model([front_img_in, back_img_in], full_outs, name="Improved_Garment_MultiTask_Model")
best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_improved_garment_model.keras")
full_end_to_end_model.save(best_ckpt_path)
print("Saved end-to-end model checkpoint to:", best_ckpt_path)

# =========================================================================
# Phase 3: Evaluate All Completed Experiments & Export CSV + JSON + Plots
# =========================================================================
csv_rows = []
# Also include the Original Notebook Baseline (Colab 4,466 split) as Reference Row
csv_rows.append({
    "experiment_id": "Ref_Notebook_Baseline",
    "description": "Original Notebook Baseline (Colab 3126/670/670, raw weights, val_loss monitor)",
    "changes_from_baseline": "None (Original reported baseline in notebook)",
    "epochs_trained": 10,
    "best_epoch": 8,
    "train_time_sec": 5579.0,
    "val_mean_macro_f1": np.nan,
    "val_condition_macro_f1": np.nan,
    "val_pilling_macro_f1": np.nan,
    "val_stains_macro_f1": np.nan,
    "val_holes_macro_f1": np.nan,
    "test_mean_macro_f1": 0.2850,
    "test_mean_accuracy": 0.4175,
    "test_condition_acc": 0.2800,
    "test_condition_macro_f1": 0.2500,
    "test_condition_weighted_f1": 0.2800,
    "test_pilling_acc": 0.2300,
    "test_pilling_macro_f1": 0.1700,
    "test_pilling_weighted_f1": 0.2800,
    "test_stains_acc": 0.5000,
    "test_stains_macro_f1": 0.4000,
    "test_stains_weighted_f1": 0.5500,
    "test_holes_acc": 0.6600,
    "test_holes_macro_f1": 0.3200,
    "test_holes_weighted_f1": 0.7400,
})

for rec in experiment_records:
    val_m = evaluate_predictions(y_val, rec["val_preds"])
    test_m = evaluate_predictions(y_test, rec["test_preds"])
    detailed_results[rec["exp_id"]] = {
        "config": {
            "description": rec["description"],
            "changes_from_baseline": rec["changes_from_baseline"],
            "epochs_trained": rec["epochs_trained"],
            "best_epoch": rec["best_epoch"],
            "train_time_sec": rec["train_time_sec"],
            "calibration_gammas": rec.get("calibration_gammas", None)
        },
        "validation": val_m,
        "test": test_m
    }
    csv_rows.append({
        "experiment_id": rec["exp_id"],
        "description": rec["description"],
        "changes_from_baseline": rec["changes_from_baseline"],
        "epochs_trained": rec["epochs_trained"],
        "best_epoch": rec["best_epoch"],
        "train_time_sec": rec["train_time_sec"],
        "val_mean_macro_f1": round(val_m["mean_macro_f1"], 4),
        "val_condition_macro_f1": round(val_m["condition"]["macro_f1"], 4),
        "val_pilling_macro_f1": round(val_m["pilling"]["macro_f1"], 4),
        "val_stains_macro_f1": round(val_m["stains"]["macro_f1"], 4),
        "val_holes_macro_f1": round(val_m["holes"]["macro_f1"], 4),
        "test_mean_macro_f1": round(test_m["mean_macro_f1"], 4),
        "test_mean_accuracy": round(float(np.mean([test_m[t]["accuracy"] for t in TASKS])), 4),
        "test_condition_acc": round(test_m["condition"]["accuracy"], 4),
        "test_condition_macro_f1": round(test_m["condition"]["macro_f1"], 4),
        "test_condition_weighted_f1": round(test_m["condition"]["weighted_f1"], 4),
        "test_pilling_acc": round(test_m["pilling"]["accuracy"], 4),
        "test_pilling_macro_f1": round(test_m["pilling"]["macro_f1"], 4),
        "test_pilling_weighted_f1": round(test_m["pilling"]["weighted_f1"], 4),
        "test_stains_acc": round(test_m["stains"]["accuracy"], 4),
        "test_stains_macro_f1": round(test_m["stains"]["macro_f1"], 4),
        "test_stains_weighted_f1": round(test_m["stains"]["weighted_f1"], 4),
        "test_holes_acc": round(test_m["holes"]["accuracy"], 4),
        "test_holes_macro_f1": round(test_m["holes"]["macro_f1"], 4),
        "test_holes_weighted_f1": round(test_m["holes"]["weighted_f1"], 4),
    })

results_df = pd.DataFrame(csv_rows)
csv_out_path = os.path.join(RESULTS_DIR, "experiment_metrics.csv")
results_df.to_csv(csv_out_path, index=False)

with open(os.path.join(RESULTS_DIR, "detailed_evaluation.json"), "w") as f:
    json.dump(detailed_results, f, indent=2)

print("\n=== SUMMARY OF ALL EXPERIMENTS ===")
print(results_df[[
    "experiment_id", "val_mean_macro_f1", "test_mean_macro_f1",
    "test_condition_macro_f1", "test_pilling_macro_f1", "test_stains_macro_f1", "test_holes_macro_f1",
    "test_condition_acc", "test_pilling_acc", "test_stains_acc", "test_holes_acc"
]].to_string(index=False))

# Select best model strictly by validation mean Macro-F1
best_rec = max(experiment_records, key=lambda r: r["val_mean_macro_f1"])
best_id = best_rec["exp_id"]
print(f"\nSelected Best Model (by Validation Mean Macro-F1): {best_id}")

# =========================================================================
# Generate Publication-Quality Plots (Confusion Matrices, Comparison Bar Charts, Curves)
# =========================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

# 1. Confusion Matrices of Best Model vs Baseline (E1) on Test Set
fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
for j, t in enumerate(TASKS):
    # Top row: E1 Baseline Reproduction
    cm_base = np.array(detailed_results["E1_Baseline_Repro"]["test"][t]["confusion_matrix"])
    sns.heatmap(
        cm_base, annot=True, fmt="d", cmap="Oranges", cbar=False,
        xticklabels=CLASS_NAMES[t], yticklabels=CLASS_NAMES[t], ax=axes[0, j]
    )
    axes[0, j].set_title(
        f"Baseline ({t.title()})\nAcc={detailed_results['E1_Baseline_Repro']['test'][t]['accuracy']:.2f} | Macro-F1={detailed_results['E1_Baseline_Repro']['test'][t]['macro_f1']:.2f}"
    )
    axes[0, j].set_xlabel("Predicted")
    axes[0, j].set_ylabel("True")

    # Bottom row: Best Selected Model
    cm_best = np.array(detailed_results[best_id]["test"][t]["confusion_matrix"])
    sns.heatmap(
        cm_best, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES[t], yticklabels=CLASS_NAMES[t], ax=axes[1, j]
    )
    axes[1, j].set_title(
        f"Improved {best_id} ({t.title()})\nAcc={detailed_results[best_id]['test'][t]['accuracy']:.2f} | Macro-F1={detailed_results[best_id]['test'][t]['macro_f1']:.2f}"
    )
    axes[1, j].set_xlabel("Predicted")
    axes[1, j].set_ylabel("True")

plt.suptitle("Test Set Confusion Matrices: Baseline (Top) vs. Best Improved Model (Bottom)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrices_comparison.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# 2. Experiment Progression Bar Chart (Validation & Test Macro-F1 across tasks)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_df = results_df[results_df["experiment_id"] != "E0_Majority"].copy()

# Left subplot: Per-task Test Macro-F1 across experiments
x_idx = np.arange(len(plot_df))
width = 0.18
colors = ["#2b5c8f", "#388e3c", "#d95f02", "#7570b3"]
for idx_t, t in enumerate(TASKS):
    vals = plot_df[f"test_{t}_macro_f1"].to_numpy()
    axes[0].bar(x_idx + (idx_t - 1.5) * width, vals, width, label=t.title(), color=colors[idx_t])

axes[0].set_xticks(x_idx)
axes[0].set_xticklabels(plot_df["experiment_id"], rotation=25, ha="right")
axes[0].set_ylabel("Test Macro-F1")
axes[0].set_title("Per-Task Test Macro-F1 Across Experiments")
axes[0].set_ylim(0, 0.65)
axes[0].legend()

# Right subplot: Mean Validation vs Test Macro-F1 & Test Accuracy
non_ref_df = plot_df[plot_df["experiment_id"] != "Ref_Notebook_Baseline"]
x_idx2 = np.arange(len(non_ref_df))
axes[1].plot(x_idx2, non_ref_df["val_mean_macro_f1"], marker="o", linewidth=2.5, label="Val Mean Macro-F1", color="#1b9e77")
axes[1].plot(x_idx2, non_ref_df["test_mean_macro_f1"], marker="s", linewidth=2.5, label="Test Mean Macro-F1", color="#d95f02")
axes[1].plot(x_idx2, non_ref_df["test_mean_accuracy"], marker="^", linewidth=2.0, linestyle="--", label="Test Mean Accuracy", color="#7570b3")
axes[1].axhline(0.2850, color="gray", linestyle=":", label="Original Notebook Baseline Macro-F1 (0.285)")
axes[1].set_xticks(x_idx2)
axes[1].set_xticklabels(non_ref_df["experiment_id"], rotation=25, ha="right")
axes[1].set_ylabel("Score")
axes[1].set_title("Overall Multi-Task Progression (Validation vs. Test)")
axes[1].set_ylim(0.20, 0.70)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "experiment_performance_comparison.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# 3. Validation Macro-F1 Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for rec in experiment_records[1:6]:
    if "history_f1" in rec and "mean_macro_f1" in rec["history_f1"]:
        ep_range = range(1, len(rec["history_f1"]["mean_macro_f1"]) + 1)
        axes[0].plot(ep_range, rec["history_f1"]["mean_macro_f1"], marker="o", linewidth=2, label=rec["exp_id"])

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation Mean Macro-F1")
axes[0].set_title("Validation Mean Macro-F1 Curves Across Experiments")
axes[0].legend()

# Right: Per-task Validation Macro-F1 curves for E4
if "history_f1" in experiment_records[4]:
    h4 = experiment_records[4]["history_f1"]
    ep_r = range(1, len(h4["mean_macro_f1"]) + 1)
    for t in TASKS:
        axes[1].plot(ep_r, h4[t], marker="s", linewidth=2, label=f"{t.title()} Val Macro-F1")
    axes[1].axvline(4.5, color="red", linestyle="--", alpha=0.6, label="Stage 2 Unfreeze")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Macro-F1")
    axes[1].set_title("E4 Per-Task Validation Macro-F1 (Warmup -> Fine-Tuning)")
    axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "validation_training_curves.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print("All plots saved to:", FIG_DIR)
