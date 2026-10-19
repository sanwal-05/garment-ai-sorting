import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = "/Users/sanjanabhowal/Desktop/DL-project"
CACHE_DIR = os.path.join(PROJECT_DIR, "data/cache")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

TASKS = ["condition", "pilling", "stains", "holes"]
NUM_CLASSES = {"condition": 5, "pilling": 5, "stains": 3, "holes": 3}
CLASS_NAMES = {
    "condition": ["1", "2", "3", "4", "5"],
    "pilling": ["1", "2", "3", "4", "5"],
    "stains": ["None", "Minor", "Major"],
    "holes": ["None", "Minor", "Major"]
}

train_data = np.load(os.path.join(CACHE_DIR, "train_features.npz"))
val_data = np.load(os.path.join(CACHE_DIR, "val_features.npz"))
test_data = np.load(os.path.join(CACHE_DIR, "test_features.npz"))
y_train = {t: train_data[t].astype(np.int32) for t in TASKS}
y_val = {t: val_data[t].astype(np.int32) for t in TASKS}
y_test = {t: test_data[t].astype(np.int32) for t in TASKS}

sw_temp = {}
for t in TASKS:
    classes = np.unique(y_train[t])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train[t])
    weights = np.minimum(np.power(weights, 0.55), 4.5)
    weights = weights / np.mean(weights[y_train[t]])
    w_map = dict(zip(classes, weights))
    sw_temp[t] = np.array([w_map[y] for y in y_train[t]], dtype=np.float32)

f_gap_in = layers.Input(shape=(1280,), name="f_gap")
b_gap_in = layers.Input(shape=(1280,), name="b_gap")
comb = layers.Concatenate(name="concat_views")([f_gap_in, b_gap_in])
comb = layers.BatchNormalization(name="bn_1")(comb)
comb = layers.Dropout(0.35, name="drop_1")(comb)
comb = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="shared_dense_256")(comb)
comb = layers.BatchNormalization(name="bn_2")(comb)
comb = layers.Dropout(0.25, name="drop_2")(comb)
outs = {t: layers.Dense(NUM_CLASSES[t], activation="softmax", name=t)(comb) for t in TASKS}
m_e2 = Model([f_gap_in, b_gap_in], outs, name="E2_TemperedWeights")

steps_per_epoch = int(np.ceil(len(y_train["condition"]) / 32))
lr_sched_e2 = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=steps_per_epoch * 20, alpha=0.05)
m_e2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched_e2),
    loss={t: "sparse_categorical_crossentropy" for t in TASKS},
    loss_weights={"condition": 1.2, "pilling": 1.2, "stains": 1.0, "holes": 1.0},
    metrics={t: ["accuracy"] for t in TASKS}
)

ds_e2_train = tf.data.Dataset.from_tensor_slices(({"f_gap": train_data["f_gap"], "b_gap": train_data["b_gap"]}, y_train, sw_temp)).shuffle(len(y_train["condition"]), seed=SEED).batch(32)

best_w = None
best_f1 = -1.0
for ep in range(17):
    m_e2.fit(ds_e2_train, epochs=1, verbose=0)
    preds = m_e2.predict({"f_gap": val_data["f_gap"], "b_gap": val_data["b_gap"]}, batch_size=64, verbose=0)
    mf1 = float(np.mean([f1_score(y_val[t], np.argmax(preds[t], axis=1), average="macro", zero_division=0) for t in TASKS]))
    if mf1 > best_f1:
        best_f1 = mf1
        best_w = m_e2.get_weights()
m_e2.set_weights(best_w)
print("Restored E2 best weights with Val Mean Macro-F1:", round(best_f1, 4))

front_input = layers.Input(shape=(224, 224, 3), name="front_image")
back_input = layers.Input(shape=(224, 224, 3), name="back_image")
backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
backbone.trainable = False

ff = layers.GlobalAveragePooling2D(name="front_gap")(backbone(front_input, training=False))
bf = layers.GlobalAveragePooling2D(name="back_gap")(backbone(back_input, training=False))
x = layers.Concatenate(name="concat_views")([ff, bf])
x = m_e2.get_layer("bn_1")(x)
x = m_e2.get_layer("drop_1")(x)
x = m_e2.get_layer("shared_dense_256")(x)
x = m_e2.get_layer("bn_2")(x)
x = m_e2.get_layer("drop_2")(x)
e2_outs = {t: m_e2.get_layer(t)(x) for t in TASKS}

full_e2_model = Model([front_input, back_input], e2_outs, name="Best_Garment_MultiTask_E2")
full_e2_model.save(os.path.join(CHECKPOINT_DIR, "best_improved_garment_model.keras"))
print("Saved Best_Garment_MultiTask_E2 to checkpoints/best_improved_garment_model.keras")

with open(os.path.join(RESULTS_DIR, "detailed_evaluation.json")) as f:
    detailed_results = json.load(f)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
for j, t in enumerate(TASKS):
    b_res = detailed_results["E1_Baseline_Repro"]["test"][t]
    cm_base = np.array(b_res["confusion_matrix"])
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Oranges", cbar=False, xticklabels=CLASS_NAMES[t], yticklabels=CLASS_NAMES[t], ax=axes[0, j])
    b_acc, b_f1 = b_res["accuracy"], b_res["macro_f1"]
    axes[0, j].set_title(f"Baseline E1 ({t.title()})\nAcc={b_acc:.2f} | Macro-F1={b_f1:.2f}")
    axes[0, j].set_xlabel("Predicted")
    axes[0, j].set_ylabel("True")

    e2_res = detailed_results["E2_Tempered_MacroF1"]["test"][t]
    cm_best = np.array(e2_res["confusion_matrix"])
    sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CLASS_NAMES[t], yticklabels=CLASS_NAMES[t], ax=axes[1, j])
    e2_acc, e2_f1 = e2_res["accuracy"], e2_res["macro_f1"]
    axes[1, j].set_title(f"Retained Improved E2 ({t.title()})\nAcc={e2_acc:.2f} | Macro-F1={e2_f1:.2f}")
    axes[1, j].set_xlabel("Predicted")
    axes[1, j].set_ylabel("True")

plt.suptitle("Test Set Confusion Matrices: Baseline E1 (Top) vs. Retained Improved Model E2 (Bottom)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrices_comparison.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Updated confusion_matrices_comparison.png")
