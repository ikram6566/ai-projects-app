#!/usr/bin/env python3
"""Batch-10 web exports: all five are classifiers -> tabular softmax in m_softmax10.json.
Data handling mirrors build_batch10.py exactly (same UCI API CSVs, same cleaning)."""
import io, json, os, urllib.request
import numpy as np
import pandas as pd

OUT = json.load(open("models.json"))
R4 = lambda x: round(float(x), 4)
def get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

os.makedirs("models", exist_ok=True)
soft = {}

def softmax_export(key, X, y, stratify=True):
    kw = {"stratify": y} if stratify else {}
    a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, **kw)
    sc = StandardScaler().fit(a)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(a), ya)
    acc = accuracy_score(yb, m.predict(sc.transform(b)))
    soft[key] = {"type": "softmax_tab", "classes": [str(c) for c in m.classes_], "features": list(X.columns),
                 "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) if v != 0 else 1.0 for v in sc.scale_],
                 "coef": [[R4(v) for v in row] for row in m.coef_],
                 "intercept": [R4(v) for v in m.intercept_],
                 "accuracy": round(float(acc), 4),
                 "defaults": {k: R4(v) for k, v in X.median().items()}}
    print(key, "softmax acc", soft[key]["accuracy"], "classes", len(soft[key]["classes"]), "features", X.shape[1])

# 46 credit approval — anonymised crx: '?' -> NaN, numeric median fill, categorical 'missing', one-hot
d = pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/27/data.csv")))
d = d.replace("?", np.nan)
num = ["A2", "A3", "A8", "A11", "A14", "A15"]
cat = [c for c in d.columns if c not in num + ["A16"]]
for c in num:
    d[c] = d[c].astype(float)
    d[c] = d[c].fillna(d[c].median())
for c in cat:
    d[c] = d[c].fillna("missing")
X = pd.get_dummies(d.drop(columns=["A16"]), columns=cat).astype(float)
softmax_export("credit", X, d["A16"])

# 47 banknote — 0 = genuine, 1 = forged
d = pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/267/data.csv")))
softmax_export("banknote", d.drop(columns=["class"]), d["class"])

# 48 page blocks — UCI id 78 (id 30 is a different dataset now)
d = pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/78/data.csv")))
softmax_export("pageblocks", d.drop(columns=["class"]), d["class"])

# 49 vehicle — drop the corrupted row (class reads '204')
d = pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/149/data.csv")))
d = d[d["class"].isin(["bus", "opel", "saab", "van"])].reset_index(drop=True)
softmax_export("vehicle", d.drop(columns=["class"]), d["class"])

# 50 yeast — target localization_site; drop the protein-name identifier column
d = pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/110/data.csv")))
drop = ["localization_site", "Sequence_Name"] + [c for c in d.columns
        if c not in ("localization_site", "Sequence_Name") and not pd.api.types.is_numeric_dtype(d[c])]
softmax_export("yeast", d.drop(columns=drop), d["localization_site"])

json.dump(soft, open("models/m_softmax10.json", "w"), separators=(",", ":"))
print("m_softmax10.json", os.path.getsize("models/m_softmax10.json") // 1024, "KB")

OUT.update(soft)
with open("models.json", "w") as f:
    json.dump(OUT, f, separators=(",", ":"))
print("models.json updated")
