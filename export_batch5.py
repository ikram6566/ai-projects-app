#!/usr/bin/env python3
"""
Batch-5 web exports: fraud (scaled LR, class-weighted), adult (reduced numeric LR),
HAR (scaled softmax + real sample windows), mushroom (decision tree), bike (linear).
Merges into models.json and writes m_tabular3.json, m_har.json, m_mushroom.json.
"""
import io, json, os, urllib.request, zipfile
import numpy as np
import pandas as pd

OUT = json.load(open("models.json"))
R4 = lambda x: round(float(x), 4)
def get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, average_precision_score, r2_score, mean_absolute_error

def scaled_lr_export(name, X_train, y_train, acc, pos, neg, defaults, extra=None):
    sc = StandardScaler().fit(X_train)
    m = LogisticRegression(max_iter=5000).fit(sc.transform(X_train), y_train)
    o = {"type": "scaled_lr", "features": list(X_train.columns),
         "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
         "coef": [R4(v) for v in m.coef_[0]], "intercept": R4(m.intercept_[0]),
         "accuracy": round(float(acc), 4), "pos": pos, "neg": neg,
         "defaults": {k: R4(v) for k, v in defaults.items()}}
    if extra: o.update(extra)
    OUT[name] = o
    print(name, "acc", round(float(acc), 4), "features", X_train.shape[1])

# ---------------------------------------------------------------- fraud
d = pd.read_csv(io.BytesIO(get("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")))
y = d["Class"].astype(int); X = d.drop(columns=["Class"])
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(a)
m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(a), ya)
prob = m.predict_proba(sc.transform(b))[:, 1]
ap = average_precision_score(yb, prob)
from sklearn.metrics import recall_score
rec = recall_score(yb, m.predict(sc.transform(b)))
o = {"type": "scaled_lr", "features": list(X.columns),
     "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
     "coef": [R4(v) for v in m.coef_[0]], "intercept": R4(m.intercept_[0]),
     "accuracy": round(float(accuracy_score(yb, m.predict(sc.transform(b)))), 4),
     "pr_auc": round(float(ap), 4), "recall": round(float(rec), 4),
     "pos": "FRAUD suspected", "neg": "Legitimate",
     "defaults": {k: R4(v) for k, v in X.median().items()}}
OUT["fraud"] = o
print("fraud pr-auc", round(float(ap), 4), "recall", round(float(rec), 4))

# ---------------------------------------------------------------- adult (reduced numeric LR)
raw = get("https://archive.ics.uci.edu/static/public/2/adult.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith("adult.data")][0]
cols = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
da = pd.read_csv(io.BytesIO(zf.read(name)), header=None, names=cols, skipinitialspace=True, na_values="?").dropna()
ya2 = (da["income"].str.contains(">50K")).astype(int)
feats = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
Xa = da[feats]
a, b, yaa, ybb = train_test_split(Xa, ya2, test_size=0.25, random_state=42, stratify=ya2)
sc = StandardScaler().fit(a)
mm = LogisticRegression(max_iter=1000).fit(sc.transform(a), yaa)
acc_a = accuracy_score(ybb, mm.predict(sc.transform(b)))
OUT["adult"] = {"type": "scaled_lr", "features": feats,
                "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
                "coef": [R4(v) for v in mm.coef_[0]], "intercept": R4(mm.intercept_[0]),
                "accuracy": round(float(acc_a), 4), "pos": "Earns over $50K", "neg": "Earns $50K or less",
                "defaults": {k: R4(v) for k, v in Xa.median().items()}}
print("adult reduced acc", round(float(acc_a), 4))

# ---------------------------------------------------------------- HAR (scaled softmax + samples)
raw = get("https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
inner = [n for n in zf.namelist() if n.endswith(".zip")][0]
zf2 = zipfile.ZipFile(io.BytesIO(zf.read(inner)))
def rd(suffix):
    n = [x for x in zf2.namelist() if x.endswith(suffix)][0]
    return zf2.read(n)
Xtr = np.loadtxt(io.BytesIO(rd("train/X_train.txt"))); ytr = np.loadtxt(io.BytesIO(rd("train/y_train.txt"))).astype(int)
Xte = np.loadtxt(io.BytesIO(rd("test/X_test.txt"))); yte = np.loadtxt(io.BytesIO(rd("test/y_test.txt"))).astype(int)
ACT = ["Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"]
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
acc_h = accuracy_score(yte, lr.predict(sc.transform(Xte)))
rng = np.random.RandomState(7)
samples = []
for cls in range(1, 7):
    idx = np.where(yte == cls)[0]
    for j in rng.choice(idx, size=2, replace=False):
        samples.append({"label": ACT[cls - 1], "x": [R4(v) for v in Xte[j]]})
OUT["har"] = {"type": "softmax_scaled", "classes": ACT,
              "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
              "coef": [[R4(v) for v in row] for row in lr.coef_],
              "intercept": [R4(v) for v in lr.intercept_],
              "accuracy": round(float(acc_h), 4), "samples": samples}
print("har acc", round(float(acc_h), 4), "samples", len(samples))

# ---------------------------------------------------------------- mushroom (decision tree)
raw = get("https://archive.ics.uci.edu/static/public/73/mushroom.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".data")][0]
mcols = ["class","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"]
dm = pd.read_csv(io.BytesIO(zf.read(name)), header=None, names=mcols)
ym = (dm["class"] == "p").astype(int)
Xm = pd.get_dummies(dm.drop(columns=["class"]), drop_first=True)
a, b, ya3, yb3 = train_test_split(Xm, ym, test_size=0.25, random_state=42, stratify=ym)
tree = DecisionTreeClassifier(max_depth=7, random_state=42).fit(a, ya3)
acc_t = accuracy_score(yb3, tree.predict(b))
t = tree.tree_
def build(node):
    if t.children_left[node] == -1:
        vals = t.value[node][0]
        return {"p": R4(vals[1] / vals.sum())}
    return {"f": int(t.feature[node]), "t": R4(t.threshold[node]),
            "l": build(int(t.children_left[node])), "r": build(int(t.children_right[node]))}
attrs = {}
for c in mcols[1:]:
    attrs[c] = sorted(dm[c].unique().tolist())
default_vec = Xm.median()
OUT["mushroom"] = {"type": "tree", "tree": build(0), "features": list(Xm.columns),
                   "attributes": attrs, "accuracy": round(float(acc_t), 4),
                   "defaults": {k: int(v) for k, v in default_vec.items()}}
print("mushroom tree acc", round(float(acc_t), 4), "features", Xm.shape[1])

# ---------------------------------------------------------------- bike (linear)
raw = get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith("hour.csv")][0]
db = pd.read_csv(io.BytesIO(zf.read(name)))
Xb = pd.get_dummies(db.drop(columns=["instant", "dteday", "casual", "registered", "cnt"]),
                    columns=["season", "weathersit", "mnth", "hr", "weekday"], drop_first=True)
yb4 = db["cnt"]
a, b, ya4, yb5 = train_test_split(Xb, yb4, test_size=0.25, random_state=42)
lin = LinearRegression().fit(a, ya4)
pv = lin.predict(b)
OUT["bike"] = {"type": "linear", "features": list(Xb.columns),
               "coef": [R4(v) for v in lin.coef_], "intercept": R4(lin.intercept_),
               "r2": round(float(r2_score(yb5, pv)), 4), "mae": round(float(mean_absolute_error(yb5, pv)), 2),
               "defaults": {k: R4(v) for k, v in Xb.median().items()}}
print("bike linear r2", OUT["bike"]["r2"], "mae", OUT["bike"]["mae"])

with open("models.json", "w") as f:
    json.dump(OUT, f, separators=(",", ":"))
print("models.json:", os.path.getsize("models.json") // 1024, "KB")
os.makedirs("models", exist_ok=True)
json.dump({"fraud": OUT["fraud"], "adult": OUT["adult"], "bike": OUT["bike"]},
          open("models/m_tabular3.json", "w"), separators=(",", ":"))
json.dump({"har": OUT["har"]}, open("models/m_har.json", "w"), separators=(",", ":"))
json.dump({"mushroom": OUT["mushroom"]}, open("models/m_mushroom.json", "w"), separators=(",", ":"))
for f in ["m_tabular3.json", "m_har.json", "m_mushroom.json"]:
    print(f, os.path.getsize(os.path.join("models", f)) // 1024, "KB")
