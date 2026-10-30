#!/usr/bin/env python3
"""Batch-9 web exports: superconductor + news as linear models (m_tabular7.json),
ionosphere + glass + waveform as tabular softmax (m_softmax9.json).
News models log1p(shares); the runner applies expm1."""
import io, json, os, urllib.request, zipfile
import numpy as np
import pandas as pd

OUT = json.load(open("models.json"))
R4 = lambda x: round(float(x), 4)
def get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

os.makedirs("models", exist_ok=True)
tab = {}   # linear models -> m_tabular7.json
soft = {}  # softmax models -> m_softmax9.json

def linear_export(key, X, y):
    a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42)
    m = LinearRegression().fit(a, ya)
    pv = m.predict(b)
    tab[key] = {"type": "linear", "features": list(X.columns),
                "coef": [R4(v) for v in m.coef_], "intercept": R4(m.intercept_),
                "r2": round(float(r2_score(yb, pv)), 4), "mae": round(float(mean_absolute_error(yb, pv)), 3),
                "defaults": {k: R4(v) for k, v in X.median().items()}}
    print(key, "linear r2", tab[key]["r2"], "mae", tab[key]["mae"], "features", X.shape[1])

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

# 41 superconductor — linear on 81 material features
raw = get("https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
d = pd.read_csv(io.BytesIO(zf.read("train.csv")))
linear_export("super", d.drop(columns=["critical_temp"]), d["critical_temp"])

# 44 news popularity — linear on log1p(shares); mae stored on the raw share scale
raw = get("https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".csv")][0]
d = pd.read_csv(io.BytesIO(zf.read(name)))
d.columns = [c.strip() for c in d.columns]
X = d.drop(columns=["url", "timedelta", "shares"])
y_log = np.log1p(d["shares"])
a, b, ya, yb, _, yb_raw = train_test_split(X, y_log, d["shares"], test_size=0.25, random_state=42)
m = LinearRegression().fit(a, ya)
pv_log = m.predict(b)
tab["news"] = {"type": "linear", "log_target": True, "features": list(X.columns),
               "coef": [R4(v) for v in m.coef_], "intercept": R4(m.intercept_),
               "r2": round(float(r2_score(yb, pv_log)), 4),
               "mae": round(float(mean_absolute_error(yb_raw, np.clip(np.expm1(pv_log), 0, None))), 1),
               "defaults": {k: R4(v) for k, v in X.median().items()}}
print("news linear r2(log)", tab["news"]["r2"], "mae(raw)", tab["news"]["mae"], "features", X.shape[1])

# 42 ionosphere — 33 usable features (2nd attribute is constant 0)
raw = get("https://archive.ics.uci.edu/static/public/52/ionosphere.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".data")][0]
d = pd.read_csv(io.BytesIO(zf.read(name)), header=None)
names = [f"pulse{(i + 2) // 2}_{'real' if i % 2 == 0 else 'imag'}" for i in range(34)]
X = d.iloc[:, :-1].drop(columns=[1]).copy(); X.columns = [n for j, n in enumerate(names) if j != 1]
softmax_export("iono", X, d.iloc[:, -1])

# 43 glass — 9 chemistry features, Id column dropped
raw = get("https://archive.ics.uci.edu/static/public/42/glass+identification.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".data")][0]
d = pd.read_csv(io.BytesIO(zf.read(name)), header=None)
X = d.iloc[:, 1:10].copy(); X.columns = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
softmax_export("glass", X, d.iloc[:, 10])

# 45 waveform — 21 attributes, 3 classes (official UCI CSV)
raw = get("https://archive.ics.uci.edu/static/public/107/data.csv")
d = pd.read_csv(io.BytesIO(raw))
softmax_export("wave", d.drop(columns=["class"]), d["class"])

json.dump(tab, open("models/m_tabular7.json", "w"), separators=(",", ":"))
print("m_tabular7.json", os.path.getsize("models/m_tabular7.json") // 1024, "KB")
json.dump(soft, open("models/m_softmax9.json", "w"), separators=(",", ":"))
print("m_softmax9.json", os.path.getsize("models/m_softmax9.json") // 1024, "KB")

OUT.update(tab); OUT.update(soft)
with open("models.json", "w") as f:
    json.dump(OUT, f, separators=(",", ":"))
print("models.json updated")
