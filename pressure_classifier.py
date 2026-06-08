"""
Pressure-Based Respiratory Disease Classifier
==============================================
Uses the expiratory pressure waveform to classify:
    Normal | Obstructive | Restrictive

Physiological basis (confirmed on balloon experiment):
  - Restrictive : highest peak pressure (stiff lung/balloon needs more P to inflate)
  - Obstructive : slowest pressure decay (airway resistance prolongs expiratory P drop)
  - Normal      : intermediate peak, moderate decay

Pipeline:
  1. Load real experimental CSVs from /experiment_data/
  2. Segment individual breath cycles (Volume resets to ~0)
  3. Reject sensor-spike breaths (|P| > SPIKE_THR)
  4. Extract 20 pressure-domain features per breath
  5. Train / test split (stratified, configurable ratio)
  6. Train Random Forest + Gradient Boosting
  7. Evaluate with classification report, confusion matrix, feature importance
  8. Save trained models to ../weights/pressure_models.pkl
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR    = "/data"
WEIGHTS_DIR = "../weights"
OUT_DIR     = os.getcwd()

DATA_FILES = {
    "Normal":      os.path.join(DATA_DIR, "normal.csv"),
    "Obstructive": os.path.join(DATA_DIR, "obsruct.csv"),
    "Restrictive": os.path.join(DATA_DIR, "resrictive.csv"),
}

SPIKE_THR      = 100.0   # cmH2O — reject breaths with |P| above this
VOL_RESET_THR  = 5.0     # mL    — volume below this marks breath boundary
MIN_SEG_LEN    = 6       # samples — discard shorter segments
TEST_SIZE      = 0.25    # 25 % held out for testing


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & BREATH SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def segment_breaths(df, label):
    """
    Split a recording into individual breath cycles.

    A breath starts when Volume crosses above VOL_RESET_THR and ends
    when it drops back below. Breaths with sensor spikes are rejected.

    Returns
    -------
    list of dicts: {pressure: np.array, volume: np.array, label: str}
    """
    vol  = df["Volume"].values
    pres = df["Pressure"].values

    reset_mask = vol <= VOL_RESET_THR
    starts = np.where(np.diff(reset_mask.astype(int)) == -1)[0]   # low→high
    ends   = np.where(np.diff(reset_mask.astype(int)) ==  1)[0]   # high→low

    breaths, n_spikes = [], 0
    for s in starts:
        next_ends = ends[ends > s]
        if len(next_ends) == 0:
            continue
        e       = next_ends[0]
        seg_p   = pres[s : e + 1]
        seg_v   = vol [s : e + 1]

        if len(seg_p) < MIN_SEG_LEN:
            continue
        if seg_p.max() > SPIKE_THR or seg_p.min() < -SPIKE_THR:
            n_spikes += 1
            continue

        breaths.append({"pressure": seg_p, "volume": seg_v, "label": label})

    return breaths, n_spikes


def load_all_breaths():
    all_breaths = []
    print("▶  Loading and segmenting breath cycles …")
    for label, path in DATA_FILES.items():
        df = pd.read_csv(path)
        breaths, n_spikes = segment_breaths(df, label)
        print(f"   {label:12s}: {len(breaths):3d} clean breaths "
              f"({n_spikes} spike(s) rejected)")
        all_breaths.extend(breaths)
    print(f"   Total: {len(all_breaths)} breaths\n")
    return all_breaths


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pressure_features(pres: np.ndarray) -> dict:
    """
    Extract 20 features from a single pressure breath segment.

    Feature groups
    ──────────────
    Peak / amplitude
      pip          peak inspiratory pressure (max P)
      peep         end-expiratory pressure   (first sample)
      driving_p    driving pressure = pip - peep
      p_auc        area under pressure curve (total work proxy)

    Expiratory decay (key discriminator for Obstructive vs Normal)
      decay_rate      linear slope of P after peak (cmH2O/sample)
      decay_half_time samples for P to fall to 50% of pip
      exp_concavity   2nd-order fit coefficient of expiratory tail
                      (positive = concave/obstructive, negative = convex)
      tau_exp         time constant of exponential fit to expiratory tail

    Shape / timing
      rise_time       samples from start to pip
      rise_slope      pip / rise_time
      insp_auc        area under inspiratory portion
      exp_auc         area under expiratory portion
      ie_auc_ratio    insp_auc / exp_auc
      p_at_half_insp  pressure at 50% of inspiratory phase

    Statistics
      mean, std, skewness, kurtosis, rms
      zero_cross_rate  (non-zero crosses through mean pressure)
    """
    n     = len(pres)
    pk_i  = int(np.argmax(pres))
    pip   = float(pres[pk_i])
    peep  = float(pres[0])
    dp    = pip - peep

    # ── inspiratory / expiratory split ───────────────────────────────────────
    insp = pres[: pk_i + 1]
    exp  = pres[pk_i :]        # includes peak sample

    # ── rise time & slope ────────────────────────────────────────────────────
    rise_time  = float(pk_i) if pk_i > 0 else 1.0
    rise_slope = dp / rise_time if rise_time > 0 else 0.0

    # ── AUC ──────────────────────────────────────────────────────────────────
    trapz    = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    p_auc    = float(trapz(np.maximum(pres, 0)))
    insp_auc = float(trapz(np.maximum(insp, 0)))
    exp_auc  = float(trapz(np.maximum(exp,  0)))
    ie_ratio = insp_auc / exp_auc if exp_auc > 1e-9 else 0.0

    # ── P at 50% of inspiratory phase ────────────────────────────────────────
    p_half_insp = float(insp[len(insp) // 2]) if len(insp) > 1 else 0.0

    # ── decay features ───────────────────────────────────────────────────────
    if len(exp) >= 3:
        x = np.arange(len(exp), dtype=float)

        # Linear decay rate (cmH2O / sample)
        decay_rate = float(np.polyfit(x, exp, 1)[0])

        # Half-time: first index where P ≤ pip/2
        half_mask = exp <= (pip / 2.0)
        decay_half = float(np.argmax(half_mask)) if half_mask.any() else float(len(exp))

        # 2nd-order concavity coefficient
        if len(exp) >= 4:
            exp_concavity = float(np.polyfit(x, exp, 2)[0])
        else:
            exp_concavity = 0.0

        # Exponential time constant τ via log-linear fit (P = pip·e^{-t/τ})
        exp_safe = np.maximum(exp, 1e-6)
        try:
            log_fit  = np.polyfit(x, np.log(exp_safe), 1)
            tau_exp  = float(-1.0 / log_fit[0]) if log_fit[0] < 0 else float(len(exp))
        except Exception:
            tau_exp = float(len(exp))
    else:
        decay_rate = decay_half = exp_concavity = tau_exp = 0.0

    # ── statistics ───────────────────────────────────────────────────────────
    mean_p = float(pres.mean())
    std_p  = float(pres.std())
    rms_p  = float(np.sqrt(np.mean(pres ** 2)))
    skew_p = float(skew(pres))
    kurt_p = float(kurtosis(pres))

    # Zero crossings through mean
    centred   = pres - mean_p
    zero_cross = float(np.sum(np.diff(np.sign(centred)) != 0) / n)

    return {
        # Peak / amplitude
        "pip":              pip,
        "peep":             peep,
        "driving_pressure": dp,
        "p_auc":            p_auc,
        # Expiratory decay
        "decay_rate":       decay_rate,
        "decay_half_time":  decay_half,
        "exp_concavity":    exp_concavity,
        "tau_exp":          tau_exp,
        # Shape / timing
        "rise_time":        rise_time,
        "rise_slope":       rise_slope,
        "insp_auc":         insp_auc,
        "exp_auc":          exp_auc,
        "ie_auc_ratio":     ie_ratio,
        "p_at_half_insp":   p_half_insp,
        # Statistics
        "p_mean":           mean_p,
        "p_std":            std_p,
        "p_skewness":       skew_p,
        "p_kurtosis":       kurt_p,
        "p_rms":            rms_p,
        "zero_cross_rate":  zero_cross,
    }


def build_feature_df(all_breaths):
    rows, labels = [], []
    for b in all_breaths:
        rows.append(extract_pressure_features(b["pressure"]))
        labels.append(b["label"])
    df = pd.DataFrame(rows)
    df["label"] = labels
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL BUILDING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_models():
    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     RandomForestClassifier(
                n_estimators=300, max_depth=None,
                min_samples_split=2, random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, random_state=42)),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                use_label_encoder=False, eval_metric="mlogloss", random_state=42)),
        ])
    return models


def train_and_evaluate(models, X_tr, X_te, y_tr, y_te, class_names):
    results = {}
    for name, model in models.items():
        print(f"\n{'='*55}\n  {name}\n{'='*55}")
        cv = cross_val_score(
            model, X_tr, y_tr,
            cv=StratifiedKFold(n_splits=min(5, min(np.bincount(y_tr))),
                               shuffle=True, random_state=42),
            scoring="f1_macro", n_jobs=-1)
        print(f"  CV F1-macro : {cv.mean():.3f} ± {cv.std():.3f}")

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)

        print("\n  Classification Report:")
        print(classification_report(y_te, y_pred, target_names=class_names))

        auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
        print(f"  ROC-AUC (macro OvR) : {auc:.3f}")

        results[name] = {"model": model, "y_pred": y_pred,
                         "y_prob": y_prob, "auc": auc}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_COLORS = {
    "Normal":      "#2ecc71",
    "Obstructive": "#e74c3c",
    "Restrictive": "#3498db",
}


def plot_pressure_waveforms(all_breaths, out_path):
    """Overlay all breath pressure waveforms per class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Pressure Waveforms per Class (all breaths)",
                 fontsize=13, fontweight="bold")

    for ax, cls in zip(axes, ["Normal", "Obstructive", "Restrictive"]):
        segs = [b["pressure"] for b in all_breaths if b["label"] == cls]
        c    = CLASS_COLORS[cls]
        for i, seg in enumerate(segs):
            ax.plot(seg, color=c, alpha=0.35, linewidth=1.0,
                    label=cls if i == 0 else "")
        # Mean waveform (pad/trim to modal length)
        modal_len = max(set(len(s) for s in segs), key=[len(s) for s in segs].count)
        trimmed   = [s[:modal_len] for s in segs if len(s) >= modal_len]
        if trimmed:
            mean_p = np.mean(trimmed, axis=0)
            ax.plot(mean_p, color=c, linewidth=2.5, label="Mean")
        ax.set_title(f"{cls}  (n={len(segs)})", fontsize=12)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Pressure (cmH₂O)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {os.path.basename(out_path)}")


def plot_feature_distributions(df_feats, out_path):
    """Box-plots of the most discriminative features across classes."""
    key_feats = ["pip", "decay_rate", "tau_exp", "decay_half_time",
                 "exp_concavity", "driving_pressure", "p_auc", "ie_auc_ratio"]
    classes = ["Normal", "Obstructive", "Restrictive"]
    colors  = [CLASS_COLORS[c] for c in classes]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Key Feature Distributions by Class",
                 fontsize=13, fontweight="bold")

    for ax, feat in zip(axes.flat, key_feats):
        data = [df_feats.loc[df_feats["label"] == c, feat].values
                for c in classes]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color="white", linewidth=2))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_title(feat, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {os.path.basename(out_path)}")


def plot_results(results, y_te, class_names, feat_cols, out_path):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 11))
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Pressure Classifier — Evaluation", fontsize=14, fontweight="bold")

    for col, (name, res) in enumerate(results.items()):
        # Confusion matrix
        cm  = confusion_matrix(y_te, res["y_pred"])
        cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)
        cmd.plot(ax=axes[0, col], colorbar=False, cmap="Blues")
        axes[0, col].set_title(f"{name}\nAUC = {res['auc']:.3f}", fontsize=11)

        # Feature importance
        clf = res["model"].named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            idx = np.argsort(imp)[-15:]
            axes[1, col].barh([feat_cols[i] for i in idx], imp[idx],
                              color="#4c9be8")
            axes[1, col].set_title("Top Feature Importances", fontsize=10)
            axes[1, col].tick_params(axis="y", labelsize=8)
        else:
            axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {os.path.basename(out_path)}")


def plot_train_test_split(y_tr, y_te, class_names, out_path):
    """Bar chart showing sample counts in train vs test per class."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"Train / Test Split  ({int((1-TEST_SIZE)*100)}% / "
                 f"{int(TEST_SIZE*100)}%)", fontsize=12, fontweight="bold")

    for ax, y, title in [(axes[0], y_tr, "Train"), (axes[1], y_te, "Test")]:
        counts = [np.sum(y == i) for i in range(len(class_names))]
        bars   = ax.bar(class_names, counts,
                        color=[CLASS_COLORS[c] for c in class_names],
                        edgecolor="white", linewidth=0.8)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom",
                    color="white", fontsize=12, fontweight="bold")
        ax.set_title(f"{title}  (n={len(y)})", fontsize=11)
        ax.set_ylabel("# Breaths")
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    fig.patch.set_facecolor("#121212")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {os.path.basename(out_path)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUT_DIR,     exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    all_breaths = load_all_breaths()

    plot_pressure_waveforms(all_breaths,
                            os.path.join(OUT_DIR, "pressure_waveforms.png"))

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    print("▶  Extracting pressure features …")
    df_feats   = build_feature_df(all_breaths)
    feat_cols  = [c for c in df_feats.columns if c != "label"]
    print(f"   {len(df_feats)} samples × {len(feat_cols)} features")
    df_feats.to_csv(os.path.join(OUT_DIR, "pressure_features.csv"), index=False)
    print("   Saved → pressure_features.csv")

    plot_feature_distributions(df_feats,
                               os.path.join(OUT_DIR, "feature_distributions.png"))

    # ── 3. Train / test split ─────────────────────────────────────────────────
    X = df_feats[feat_cols].values
    le = LabelEncoder()
    y  = le.fit_transform(df_feats["label"].values)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

    print(f"\n▶  Split: {len(X_tr)} train / {len(X_te)} test "
          f"(stratified {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})")
    for i, cls in enumerate(le.classes_):
        print(f"   {cls:12s}: {np.sum(y_tr==i)} train / {np.sum(y_te==i)} test")

    plot_train_test_split(y_tr, y_te, le.classes_,
                          os.path.join(OUT_DIR, "train_test_split.png"))

    # ── 4. Train & evaluate ───────────────────────────────────────────────────
    print("\n▶  Training models …")
    models  = build_models()
    results = train_and_evaluate(models, X_tr, X_te, y_tr, y_te, le.classes_)

    # ── 5. Result plots ───────────────────────────────────────────────────────
    print("\n▶  Plotting results …")
    plot_results(results, y_te, le.classes_, feat_cols,
                 os.path.join(OUT_DIR, "pressure_model_results.png"))

    # ── 6. Save weights ───────────────────────────────────────────────────────
    # Pick the model with best test AUC
    best_name = max(results, key=lambda n: results[n]["auc"])
    weights = {
        "models":        {n: r["model"] for n, r in results.items()},
        "label_encoder": le,
        "feature_cols":  feat_cols,
        "best_model":    best_name,
    }
    weights_path = os.path.join(WEIGHTS_DIR, "pressure_models.pkl")
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)

    print(f"\n✅  Done.")
    print(f"    Best model  : {best_name}  (AUC = {results[best_name]['auc']:.3f})")
    print(f"    Weights     : {weights_path}")
    print(f"    Output files:")
    for fname in ["pressure_waveforms.png", "feature_distributions.png",
                  "train_test_split.png", "pressure_model_results.png",
                  "pressure_features.csv"]:
        print(f"      • {fname}")