import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to this file so it works from any working directory
BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ============================ Helper functions ============================
def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.exp(-np.logaddexp(0, -z))


def log_loss_np(X_aug, y, beta):
    """Mean binary cross-entropy, computed stably (never takes log of zero)."""
    z = X_aug @ beta
    return np.mean(np.logaddexp(0, z) - y * z)


def add_intercept(X):
    """Prepend a column of ones for the intercept."""
    return np.column_stack([np.ones(len(X)), X])


def train_logreg(X, y, lr=0.1, epochs=20000):
    """Full-batch gradient descent for binary logistic regression (NumPy only).

    Returns the final coefficients [beta_0, beta_1, beta_2] and the loss
    history: the loss before training plus the loss after each update
    (epochs + 1 values in total).
    """
    X_aug = add_intercept(X)
    beta = np.zeros(X_aug.shape[1])          
    losses = [log_loss_np(X_aug, y, beta)]  
    for _ in range(epochs):
        p = sigmoid(X_aug @ beta)
        grad = X_aug.T @ (p - y) / len(y)   
        beta = beta - lr * grad              
        losses.append(log_loss_np(X_aug, y, beta))
    return beta, np.array(losses)


def evaluate(y_true, p, thr):
    """Confusion matrix and metrics at a threshold. Log loss uses probabilities."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(
        threshold=thr, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        accuracy=accuracy_score(y_true, pred),
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        log_loss=log_loss(y_true, p, labels=[0, 1]),
    )


# ===================== Part 1: four hand-set cases =====================
print("=== Part 1: score, probability, and predicted class ===")
cases = {
    "A": (-1.0, 0.8, 2.0, 0.50),
    "B": (-1.0, 0.8, 2.0, 0.70),
    "C": (-1.0, -0.8, 2.0, 0.50),
    "D": (1.0, 0.0, 2.0, 0.50),
}
for name, (b0, b1, x, thr) in cases.items():
    z = b0 + b1 * x
    p = sigmoid(z)
    print(f"Case {name}: z = {z:5.2f}   p = {p:.4f}   "
          f"threshold = {thr:.2f}   predicted class = {int(p >= thr)}")


# ============ Part 2: data, leakage-free split, and scaling ============
print("\n=== Part 2: data preparation ===")
data = load_breast_cancer()
names = ["mean radius", "mean texture"]  # exactly these two features, in this order
idx = [list(data.feature_names).index(n) for n in names]
X = data.data[:, idx]
y = (data.target == 0).astype(int)       # 1 = malignant (positive), 0 = benign

# Split row indices: development/test first, then training/validation
all_idx = np.arange(len(y))
dev_idx, test_idx = train_test_split(
    all_idx, test_size=0.20, stratify=y, random_state=42)
train_idx, val_idx = train_test_split(
    dev_idx, test_size=0.25, stratify=y[dev_idx], random_state=42)

# Fit the scaler on training rows only, then apply it to every partition
scaler = StandardScaler().fit(X[train_idx])
X_train = scaler.transform(X[train_idx])
X_val = scaler.transform(X[val_idx])
X_test = scaler.transform(X[test_idx])
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

print("Features:", names)
print("Missing values by feature:")
for name, count in zip(names, np.isnan(X).sum(axis=0)):
    print(f"  {name}: {int(count)}")
print("Target coding: 0 = benign (negative), 1 = malignant (positive)")
print("Overall class counts: benign(0) =", int((y == 0).sum()),
      ", malignant(1) =", int((y == 1).sum()))
for label, ids in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
    c = np.bincount(y[ids], minlength=2)
    print(f"{label:10s} rows={len(ids):4d}  benign(0)={c[0]:3d}  malignant(1)={c[1]:3d}")

# Verify the three index sets do not overlap and together cover every row
s_tr, s_va, s_te = set(train_idx), set(val_idx), set(test_idx)
assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
assert len(s_tr | s_va | s_te) == len(y)
print("Index sets are disjoint and cover all", len(y), "rows.")


# ============== Part 3: NumPy logistic regression training ==============
print("\n=== Part 3: NumPy gradient descent ===")
beta, losses = train_logreg(X_train, y_train, lr=0.1, epochs=20000)
print("Recorded loss values:", len(losses))
print(f"Initial training loss: {losses[0]:.6f}")
print(f"Final training loss:   {losses[-1]:.6f}")
print("Learned coefficients (standardized units):")
print(f"  intercept (beta_0):    {beta[0]:.6f}")
print(f"  mean radius (beta_1):  {beta[1]:.6f}")
print(f"  mean texture (beta_2): {beta[2]:.6f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(len(losses)), losses)
ax.set_title("NumPy logistic regression: training loss vs. epoch")
ax.set_xlabel("Epoch (0 = before training)")
ax.set_ylabel("Mean binary cross-entropy (training)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(REPORT_DIR / "training_loss.png", dpi=150)
plt.close(fig)


# =============== Part 4: scikit-learn reference and baseline ===============
print("\n=== Part 4: scikit-learn reference and majority baseline ===")
sk = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000, tol=1e-9)
sk.fit(X_train, y_train)  # sklearn fits its own intercept: do not add a column of ones
base = DummyClassifier(strategy="prior").fit(X_train, y_train)

# Confirm that column 1 of predict_proba is the probability of class 1 (malignant)
assert list(sk.classes_) == [0, 1] and list(base.classes_) == [0, 1]


def class1_probs(features):
    """Probability of class 1 (malignant) from each model."""
    return {
        "numpy": sigmoid(add_intercept(features) @ beta),
        "sklearn": sk.predict_proba(features)[:, 1],
        "baseline": base.predict_proba(features)[:, 1],
    }


P = {"validation": class1_probs(X_val), "test": class1_probs(X_test)}
y_of = {"validation": y_val, "test": y_test}

beta_sk = np.concatenate([sk.intercept_, sk.coef_[0]])
print("Coefficients [intercept, mean radius, mean texture] (standardized units)")
print("  NumPy  :", np.round(beta, 6))
print("  sklearn:", np.round(beta_sk, 6))
print("Mean absolute difference of validation probabilities (NumPy vs sklearn):",
      np.mean(np.abs(P["validation"]["numpy"] - P["validation"]["sklearn"])))
print("Baseline P(malignant) learned from training data:",
      round(float(P["validation"]["baseline"][0]), 6))


# ======= Part 5: select a threshold (validation only), then evaluate =======
print("\n=== Part 5: threshold selection and evaluation ===")
thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
thr_rows = [evaluate(y_val, P["validation"]["numpy"], t) for t in thresholds]

# Highest validation F1; ties go to the smallest threshold
best_f1 = max(r["f1"] for r in thr_rows)
selected = min(r["threshold"] for r in thr_rows if np.isclose(r["f1"], best_f1))

print("NumPy model, validation set:")
for r in thr_rows:
    print(f"  threshold={r['threshold']:.2f}  FP={r['fp']:3d}  FN={r['fn']:3d}  "
          f"precision={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1']:.4f}")
print("Selected threshold (highest validation F1, smallest on ties):", selected)

# Models, scaler, and threshold are now frozen. Evaluate validation and test.
configs = [
    ("baseline", "default", 0.50),
    ("numpy", "default", 0.50),
    ("numpy", "selected", selected),
    ("sklearn", "default", 0.50),
    ("sklearn", "selected", selected),  # same threshold, chosen from the NumPy model
]
rows = []
for split in ["validation", "test"]:
    for model, policy, thr in configs:
        r = evaluate(y_of[split], P[split][model], thr)
        rows.append({"split": split, "model": model, "policy": policy, **r})

print("\nConfusion matrices: rows = actual class, columns = predicted class "
      "(0 = benign, 1 = malignant)")
print("split       model     policy    thr   TN  FP  FN  TP   "
      "acc     prec    recall  f1      logloss")
for r in rows:
    print(f"{r['split']:11s} {r['model']:9s} {r['policy']:9s} {r['threshold']:.2f} "
          f"{r['tn']:3d} {r['fp']:3d} {r['fn']:3d} {r['tp']:3d}  "
          f"{r['accuracy']:.4f}  {r['precision']:.4f}  {r['recall']:.4f}  "
          f"{r['f1']:.4f}  {r['log_loss']:.4f}")


# ========================== Part 6: export CSVs ==========================
def fmt(value):
    """Keep integers and strings as-is; write metrics with 8 decimal places."""
    return value if isinstance(value, (str, int)) else f"{value:.8f}"


thr_cols = ["threshold", "tn", "fp", "fn", "tp",
            "accuracy", "precision", "recall", "f1"]
with open(REPORT_DIR / "threshold_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=thr_cols)
    writer.writeheader()
    for r in thr_rows:
        writer.writerow({k: fmt(r[k]) for k in thr_cols})

all_cols = ["split", "model", "policy", "threshold", "tn", "fp", "fn", "tp",
            "accuracy", "precision", "recall", "f1", "log_loss"]
with open(REPORT_DIR / "assignment_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_cols)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: fmt(r[k]) for k in all_cols})

print("\nSaved threshold_results.csv, assignment_results.csv, and "
      "training_loss.png to:", REPORT_DIR)