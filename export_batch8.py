#!/usr/bin/env python3
"""Batch-8 web exports: cover (10 continuous features) + steel as tabular softmax,
abalone + turbine as linear models, letter as tabular softmax."""
import io, json, os, urllib.request, zipfile, gzip
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
tab = {}   # linear models -> m_tabular6.json
soft = {}  # softmax models -> m_softmax8.json

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

# 36 cover — browser model uses the 10 continuous terrain features only
raw = get("https://archive.ics.uci.edu/static/public/31/covertype.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if "covtype.data" in n][0]
d = pd.read_csv(io.BytesIO(gzip.decompress(zf.read(name))), header=None)
cont_names = ["Elevation", "Aspect", "Slope", "HorizDistToHydrology", "VertDistToHydrology",
              "HorizDistToRoadways", "Hillshade9am", "HillshadeNoon", "Hillshade3pm", "HorizDistToFirePoints"]
X = d.iloc[:, :10].copy(); X.columns = cont_names
y = d.iloc[:, -1].astype(str)
Xs, _, ys, _ = train_test_split(X, y, train_size=120000, random_state=42, stratify=y)
softmax_export("cover", Xs, ys)

# 37 steel — all 27 measurements, 7 fault types
raw = get("https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".NNA")][0]
var_names = [l.strip() for l in zf.read("Faults27x7_var").decode().splitlines() if l.strip()]
d = pd.read_csv(io.BytesIO(zf.read(name)), sep=r"\s+", header=None)
fault_names = var_names[27:34]
feat_names = var_names[:27]
y = d.iloc[:, 27:34].idxmax(axis=1).map(lambda i: fault_names[i - 27])
X = d.iloc[:, :27].copy(); X.columns = feat_names
softmax_export("steel", X, y)

# 38 abalone — linear on physical measurements (sex one-hot)
raw = get("https://archive.ics.uci.edu/static/public/1/abalone.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".data")][0]
cols = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
d = pd.read_csv(io.BytesIO(zf.read(name)), header=None, names=cols)
X = pd.get_dummies(d.drop(columns=["rings"]), columns=["sex"], drop_first=True).astype(float)
linear_export("abalone", X, d["rings"])

# 39 turbine — linear on process sensors (drop year so the demo is about the sensors)
raw = get("https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist() if n.endswith(".csv")]
d = pd.concat(frames, ignore_index=True)
X = d.drop(columns=["CO", "NOX"])
linear_export("turbine", X, d["CO"])

# 40 letter — 16 features, 26 classes
raw = get("https://archive.ics.uci.edu/static/public/59/letter+recognition.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".data")][0]
d = pd.read_csv(io.BytesIO(zf.read(name)), header=None)
feat = ["x_box", "y_box", "width", "height", "onpix", "x_bar", "y_bar", "x2bar", "y2bar",
        "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx"]
X = d.iloc[:, 1:].copy(); X.columns = feat
softmax_export("letter", X, d.iloc[:, 0])

json.dump(tab, open("models/m_tabular6.json", "w"), separators=(",", ":"))
print("m_tabular6.json", os.path.getsize("models/m_tabular6.json") // 1024, "KB")
json.dump(soft, open("models/m_softmax8.json", "w"), separators=(",", ":"))
print("m_softmax8.json", os.path.getsize("models/m_softmax8.json") // 1024, "KB")

OUT.update(tab); OUT.update(soft)
with open("models.json", "w") as f:
    json.dump(OUT, f, separators=(",", ":"))
print("models.json updated")
