#!/usr/bin/env python3
"""
Batch-4 model exports for the AI Projects App: bank marketing (scaled LR),
AG News (TF-IDF softmax), movie recommender (precomputed top-5 lists),
stock next-day return (linear regression), CIFAR-10 (small CNN weights).

Merges the new models into models.json and writes NEW split files only
(m_tabular2.json, m_agnews.json, m_reco.json, m_cifar_meta.json, m_cifar_N.json).
Run with the portfolio venv python (has tensorflow + sklearn).
"""
import io, json, os, urllib.request, zipfile
import numpy as np
import pandas as pd

OUT = json.load(open("models.json"))
R4 = lambda x: round(float(x), 4)
def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# ---------------------------------------------------------------- bank (scaled LR)
def export_scaled_lr(name, X_train, y_train, acc, pos, neg, defaults):
    sc = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=5000).fit(sc.transform(X_train), y_train)
    OUT[name] = {
        "type": "scaled_lr", "features": list(X_train.columns),
        "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
        "coef": [R4(v) for v in model.coef_[0]], "intercept": R4(model.intercept_[0]),
        "accuracy": round(float(acc), 4), "pos": pos, "neg": neg,
        "defaults": {k: R4(v) for k, v in defaults.items()},
    }
    print(name, "acc", round(float(acc), 4), "features", X_train.shape[1])

raw = get("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
def find_csv(z, target):
    for n in z.namelist():
        if n.endswith(target): return z.read(n)
        if n.endswith(".zip"):
            try:
                r = find_csv(zipfile.ZipFile(io.BytesIO(z.read(n))), target)
                if r: return r
            except Exception: pass
d = pd.read_csv(io.BytesIO(find_csv(zf, "bank-full.csv")), sep=";")
y = (d["y"] == "yes").astype(int)
X = pd.get_dummies(d.drop(columns=["y"]), drop_first=True)
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(a)
m = LogisticRegression(max_iter=1000).fit(sc.transform(a), ya)
export_scaled_lr("bank", a, ya, accuracy_score(yb, m.predict(sc.transform(b))),
                 "Likely to SUBSCRIBE", "Unlikely to subscribe", X.median())

# ---------------------------------------------------------------- AG News (TF-IDF softmax)
from sklearn.feature_extraction.text import TfidfVectorizer
tr = pd.read_parquet("https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/train-00000-of-00001.parquet")
te = pd.read_parquet("https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet")
vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform(tr["text"]); Xte = vec.transform(te["text"])
m = LogisticRegression(max_iter=1000).fit(Xtr, tr["label"])
acc = accuracy_score(te["label"], m.predict(Xte))
vocab = vec.vocabulary_; idf = vec.idf_; coef = m.coef_  # (4, V)
strength = np.max(np.abs(coef), axis=0)
terms = sorted(vocab.items(), key=lambda kv: -strength[kv[1]])[:5000]
OUT["agnews"] = {
    "type": "tfidf_softmax",
    "vocab": {t: [R4(idf[i])] + [R4(coef[k][i]) for k in range(4)] for t, i in terms},
    "intercept": [R4(v) for v in m.intercept_],
    "classes": ["World", "Sports", "Business", "Sci/Tech"],
    "accuracy": round(float(acc), 4),
}
print("agnews acc", round(float(acc), 4), "vocab exported", len(terms))

# ---------------------------------------------------------------- recommender (precomputed)
r = pd.read_csv(io.StringIO(get("https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat").decode("utf-8", "replace")),
                sep="::", engine="python", header=None, names=["user", "item", "rating", "ts"])
tu, ti = 50, 50
for _ in range(5):
    uc = r["user"].value_counts(); ic = r["item"].value_counts()
    keep_u, keep_i = uc[uc >= tu].index, ic[ic >= ti].index
    if len(keep_u) <= 6000 and len(keep_i) <= 4000: break
    tu = int(tu * 1.5); ti = int(ti * 1.5)
ratings = r[r["user"].isin(keep_u) & r["item"].isin(keep_i)].copy()
try:
    mv = pd.read_csv(io.StringIO(get("https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat").decode("utf-8", "replace")),
                     sep="::", engine="python", header=None, names=["item", "title", "genres"])
    titles = dict(zip(mv["item"], mv["title"]))
except Exception:
    titles = {}
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
rtr, rte = train_test_split(ratings, test_size=0.2, random_state=42)
users = sorted(ratings["user"].unique()); items = sorted(ratings["item"].unique())
uidx = {u: k for k, u in enumerate(users)}; iidx = {i: k for k, i in enumerate(items)}
R = np.zeros((len(users), len(items)))
for row in rtr.itertuples():
    R[uidx[row.user], iidx[row.item]] = row.rating
means = np.array([R[k][R[k] > 0].mean() if (R[k] > 0).any() else ratings["rating"].mean() for k in range(len(users))])
Rc = np.where(R > 0, R - means[:, None], 0)
svd = TruncatedSVD(n_components=30, random_state=42)
pred_all = (svd.fit_transform(Rc) @ svd.components_) + means[:, None]
preds = [float(np.clip(pred_all[uidx[row.user], iidx[row.item]], 0, 10)) for row in rte.itertuples()]
rmse = float(np.sqrt(mean_squared_error(list(rte["rating"]), preds)))
sample_users = list(r["user"].value_counts().head(12).index)
reco_users = []
for u in sample_users:
    k = uidx[u]
    seen_items = set(rtr[rtr["user"] == u]["item"])
    favs = sorted([(titles.get(row.item, f"Movie #{row.item}"), int(row.rating)) for row in rtr[rtr["user"] == u].itertuples()],
                  key=lambda t: -t[1])[:3]
    cands = [(items[j], float(pred_all[k, j])) for j in range(len(items)) if items[j] not in seen_items]
    top5 = sorted(cands, key=lambda t: -t[1])[:5]
    reco_users.append({"id": int(u), "favorites": favs,
                       "recommendations": [[str(titles.get(i, f"Movie #{i}")), round(p, 2)] for i, p in top5]})
OUT["reco"] = {"type": "reco", "rmse": round(rmse, 4), "users": reco_users}
print("reco rmse", round(rmse, 4), "sample users", len(reco_users))

# ---------------------------------------------------------------- stock (return regression)
j = json.loads(get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=10y&interval=1d"))
res0 = j["chart"]["result"][0]; q = res0["indicators"]["quote"][0]
s = pd.DataFrame({"Date": pd.to_datetime(res0["timestamp"], unit="s"), "Close": q["close"]}).dropna().sort_values("Date").reset_index(drop=True)
s["ret1"] = s["Close"].pct_change(1); s["ret5"] = s["Close"].pct_change(5); s["ret10"] = s["Close"].pct_change(10)
s["ma10"] = s["Close"].rolling(10).mean() / s["Close"] - 1
s["ma30"] = s["Close"].rolling(30).mean() / s["Close"] - 1
s["vol10"] = s["ret1"].rolling(10).std()
s["target_ret"] = s["Close"].pct_change().shift(-1)
s2 = s.dropna().reset_index(drop=True)
feats = ["ret1", "ret5", "ret10", "ma10", "ma30", "vol10"]
cut = int(len(s2) * 0.8)
Xtr_s, Xte_s = s2[feats][:cut], s2[feats][cut:][:-1]
ytr_s = s2["target_ret"][:cut]
today_close = s2["Close"][cut:].values[:-1]
actual_next = s2["Close"].shift(-1)[cut:].values[:-1]
actual_ret = s2["target_ret"][cut:].values[:-1]
lm = LinearRegression().fit(Xtr_s, ytr_s)
pr = lm.predict(Xte_s)
mae = float(mean_absolute_error(actual_next, today_close * (1 + pr)))
dir_acc = float(np.mean(np.sign(pr) == np.sign(actual_ret)))
naive = float(np.mean(np.abs(actual_next - today_close)))
OUT["stock"] = {"type": "stock_ret", "features": feats,
                "coef": [R4(v) for v in lm.coef_], "intercept": R4(lm.intercept_),
                "mae": round(mae, 2), "naive_mae": round(naive, 2), "dir_acc": round(dir_acc, 4),
                "defaults": {k: R4(v) for k, v in s2[feats].median().items()},
                "last_close": R4(s["Close"].iloc[-1])}
print("stock mae", round(mae, 2), "naive", round(naive, 2), "dir", round(dir_acc, 4))

# ---------------------------------------------------------------- CIFAR-10 CNN
import tensorflow as tf
(xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
X_train = xtr[:20000] / 255.0; y_tr = ytr[:20000].ravel()
X_test = xte / 255.0; y_te = yte.ravel()
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_tr, epochs=8, batch_size=128, verbose=0)
loss, acc = model.evaluate(X_test, y_te, verbose=0)
print("cifar acc", round(float(acc), 4))
weights = model.get_weights()  # conv1W, conv1b, conv2W, conv2b, d1W, d1b, d2W, d2b
flat = np.concatenate([w.ravel() for w in weights])
R3 = lambda arr: np.round(arr, 3)
flat_r = R3(flat)
OUT["cifar"] = {"type": "cnn", "accuracy": round(float(acc), 4),
                "labels": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                "weights": flat_r.tolist()}
print("cifar flat weights:", len(flat_r))

with open("models.json", "w") as f:
    json.dump(OUT, f, separators=(",", ":"))
print("models.json updated:", os.path.getsize("models.json") // 1024, "KB")

# ---------------------------------------------------------------- split files (new ones only)
os.makedirs("models", exist_ok=True)
json.dump({"bank": OUT["bank"], "stock": OUT["stock"]}, open("models/m_tabular2.json", "w"), separators=(",", ":"))
def trunc_softmax(model_, n):
    items = sorted(model_["vocab"].items(), key=lambda kv: -max(abs(v) for v in kv[1][1:]))[:n]
    o = dict(model_); o["vocab"] = dict(items); return o
json.dump({"agnews": trunc_softmax(OUT["agnews"], 2600)}, open("models/m_agnews.json", "w"), separators=(",", ":"))
json.dump({"reco": OUT["reco"]}, open("models/m_reco.json", "w"), separators=(",", ":"))
meta = {k: OUT["cifar"][k] for k in ["type", "accuracy", "labels"]}
json.dump({"cifar_meta": meta}, open("models/m_cifar_meta.json", "w"), separators=(",", ":"))
W = OUT["cifar"]["weights"]
chunk = 14000
n_chunks = 0
for i in range(0, len(W), chunk):
    json.dump({"cifarChunk": W[i:i + chunk]}, open(f"models/m_cifar_{i // chunk}.json", "w"), separators=(",", ":"))
    n_chunks += 1
print("cifar chunks:", n_chunks)
for f in sorted(os.listdir("models")):
    print(f, os.path.getsize(os.path.join("models", f)) // 1024, "KB")
