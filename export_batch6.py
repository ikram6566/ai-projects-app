#!/usr/bin/env python3
"""
Batch-6 web exports: shoppers (scaled LR), concrete (linear), parkinsons (scaled LR),
newsgroups (TF-IDF softmax, truncated vocab), MNIST CNN (flat weights, chunked).
"""
import io, json, os, urllib.request, zipfile
import numpy as np
import pandas as pd

OUT = json.load(open("models.json"))
R4 = lambda x: round(float(x), 4)
def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------- shoppers (scaled LR)
raw = get("https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".csv")][0]
d = pd.read_csv(io.BytesIO(zf.read(name)))
y = d["Revenue"].astype(int)
X = pd.get_dummies(d.drop(columns=["Revenue"]), columns=["Month", "VisitorType", "Weekend"], drop_first=True)
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(a)
m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(a), ya)
acc = accuracy_score(yb, m.predict(sc.transform(b)))
OUT["shoppers"] = {"type": "scaled_lr", "features": list(X.columns),
                    "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
                    "coef": [R4(v) for v in m.coef_[0]], "intercept": R4(m.intercept_[0]),
                    "accuracy": round(float(acc), 4), "pos": "Will purchase", "neg": "Will leave without buying",
                    "defaults": {k: R4(v) for k, v in X.median().items()}}
print("shoppers LR acc", round(float(acc), 4), "features", X.shape[1])

# ---------------------------------------------------------------- concrete (linear)
raw = get("https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".xls")][0]
d = pd.read_excel(io.BytesIO(zf.read(name)))
d.columns = ["cement", "slag", "flyash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age", "strength"]
X, y = d.drop(columns=["strength"]), d["strength"]
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42)
lin = LinearRegression().fit(a, ya)
pv = lin.predict(b)
OUT["concrete"] = {"type": "linear", "features": list(X.columns),
                   "coef": [R4(v) for v in lin.coef_], "intercept": R4(lin.intercept_),
                   "r2": round(float(r2_score(yb, pv)), 4), "mae": round(float(mean_absolute_error(yb, pv)), 2),
                   "defaults": {k: R4(v) for k, v in X.median().items()}}
print("concrete linear r2", OUT["concrete"]["r2"])

# ---------------------------------------------------------------- parkinsons (scaled LR)
raw = get("https://archive.ics.uci.edu/static/public/174/parkinsons.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".data")][0]
d = pd.read_csv(io.BytesIO(zf.read(name)))
y = d["status"]; X = d.drop(columns=["name", "status"])
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(a)
m = LogisticRegression(max_iter=1000).fit(sc.transform(a), ya)
acc = accuracy_score(yb, m.predict(sc.transform(b)))
OUT["parkinsons"] = {"type": "scaled_lr", "features": list(X.columns),
                      "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
                      "coef": [R4(v) for v in m.coef_[0]], "intercept": R4(m.intercept_[0]),
                      "accuracy": round(float(acc), 4),
                      "pos": "Pattern matches Parkinson's group", "neg": "Pattern matches healthy group",
                      "defaults": {k: R4(v) for k, v in X.median().items()}}
print("parkinsons acc", round(float(acc), 4))

json.dump({"shoppers": OUT["shoppers"], "concrete": OUT["concrete"], "parkinsons": OUT["parkinsons"]},
          open("models/m_tabular4.json", "w"), separators=(",", ":"))
print("m_tabular4.json", os.path.getsize("models/m_tabular4.json") // 1024, "KB")

# ---------------------------------------------------------------- newsgroups (tfidf softmax)
from sklearn.datasets import fetch_20newsgroups
tr = fetch_20newsgroups(subset="train", random_state=42)
te = fetch_20newsgroups(subset="test", random_state=42)
vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform(tr.data)
lr = LogisticRegression(max_iter=1000).fit(Xtr, tr.target)
acc = accuracy_score(te.target, lr.predict(vec.transform(te.data)))
vocab = vec.get_feature_names_out()
idf = vec.idf_
importance = np.abs(lr.coef_).max(axis=0)
keep = np.argsort(-importance)[:1800]
keep_sorted = np.sort(keep)
terms = [vocab[i] for i in keep_sorted]
OUT["newsgroups"] = {"type": "tfidf_softmax", "classes": list(tr.target_names),
                     "vocab": {t: j for j, t in enumerate(terms)},
                     "idf": [R4(idf[i]) for i in keep_sorted],
                     "coef": [[R4(lr.coef_[k][i]) for i in keep_sorted] for k in range(20)],
                     "intercept": [R4(v) for v in lr.intercept_],
                     "accuracy": round(float(acc), 4)}
json.dump({"newsgroups": OUT["newsgroups"]}, open("models/m_newsgroups.json", "w"), separators=(",", ":"))
print("newsgroups acc", round(float(acc), 4), "file", os.path.getsize("models/m_newsgroups.json") // 1024, "KB")

# ---------------------------------------------------------------- MNIST CNN
import tensorflow as tf
tf.random.set_seed(42)
(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr = xtr.reshape(-1, 28, 28, 1) / 255.0; xte = xte.reshape(-1, 28, 28, 1) / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(xtr, ytr, epochs=5, batch_size=128, verbose=0, validation_split=0.05)
loss, acc_c = model.evaluate(xte, yte, verbose=0)
print("mnist cnn acc", round(float(acc_c), 4))
flat = []
for w in model.get_weights():
    flat.extend(np.round(w.astype(np.float64).ravel(), 3).tolist())
print("mnist cnn weights:", len(flat))
json.dump({"mnistcnn_meta": {"accuracy": round(float(acc_c), 4), "n_weights": len(flat)}},
          open("models/m_mnistcnn_meta.json", "w"), separators=(",", ":"))
CH = 12000
for i in range(0, len(flat), CH):
    json.dump({"mnistcnn_w" + str(i // CH): flat[i:i + CH]},
              open(f"models/m_mnistcnn_{i // CH}.json", "w"), separators=(",", ":"))
print("mnistcnn chunks:", (len(flat) + CH - 1) // CH)

with open("models.json", "w") as f:
    json.dump(OUT, f, separators=(",", ":"))
print("models.json:", os.path.getsize("models.json") // 1024, "KB")
