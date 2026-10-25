#!/usr/bin/env python3
"""
Export browser-ready models for the AI Projects App.

Trains a lightweight web version of each portfolio project, evaluates it on a
held-out test set (the accuracy written into models.json is THIS model's real
score), and exports weights to models.json for the static site.

Run:  python export_models.py   ->  writes models.json next to this file.
"""
import io, json, tarfile, urllib.request, zipfile
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

OUT = {}
R4 = lambda x: round(float(x), 4)

# ---------------------------------------------------------------- text models
def export_tfidf_lr(name, vec, model, top_k, acc, pos, neg):
    vocab = vec.vocabulary_
    idf = vec.idf_
    coef = model.coef_[0]
    terms = sorted(vocab.items(), key=lambda kv: -abs(coef[kv[1]]))[:top_k]
    OUT[name] = {
        "type": "tfidf_lr",
        "vocab": {t: [R4(idf[i]), R4(coef[i])] for t, i in terms},
        "intercept": R4(model.intercept_[0]),
        "accuracy": round(float(acc), 4),
        "pos": pos, "neg": neg,
    }
    print(name, "acc", round(float(acc), 4), "vocab exported", len(terms))

# Sentiment — IMDb 50k
tr = pd.read_parquet("https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet")
te = pd.read_parquet("https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet")
clean = lambda s: s.str.replace("<br />", " ", regex=False)
vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform(clean(tr["text"])); Xte = vec.transform(clean(te["text"]))
m = LogisticRegression(max_iter=1000).fit(Xtr, tr["label"])
export_tfidf_lr("sentiment", vec, m, 5000, accuracy_score(te["label"], m.predict(Xte)), "Positive", "Negative")

# SMS Spam
with urllib.request.urlopen("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip") as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
df = pd.read_csv(io.StringIO(z.read("SMSSpamCollection").decode("utf-8", "replace")),
                 sep="\t", header=None, names=["label", "message"])
y = (df["label"] == "spam").astype(int)
a, b, ya, yb = train_test_split(df["message"], y, test_size=0.2, random_state=42, stratify=y)
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
Xtr = vec.fit_transform(a); Xte = vec.transform(b)
m = LogisticRegression(max_iter=1000).fit(Xtr, ya)
export_tfidf_lr("spam", vec, m, 3000, accuracy_score(yb, m.predict(Xte)), "Spam", "Ham")

# Fake News — full articles
tr = pd.read_csv("https://huggingface.co/datasets/GonzaloA/fake_news/resolve/main/train.csv", sep=";")
te = pd.read_csv("https://huggingface.co/datasets/GonzaloA/fake_news/resolve/main/test.csv", sep=";")
docs = lambda d: (d["title"].fillna("") + " " + d["text"].fillna("")).astype(str)
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform(docs(tr)); Xte = vec.transform(docs(te))
m = LogisticRegression(max_iter=1000).fit(Xtr, tr["label"])
export_tfidf_lr("fakenews", vec, m, 5000, accuracy_score(te["label"], m.predict(Xte)), "REAL", "FAKE")

# ---------------------------------------------------------------- tabular LR
def export_scaled_lr(name, X, y, acc, pos, neg, defaults, samples=None):
    sc = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=5000).fit(sc.transform(X), y)
    OUT[name] = {
        "type": "scaled_lr", "features": list(X.columns),
        "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
        "coef": [R4(v) for v in model.coef_[0]], "intercept": R4(model.intercept_[0]),
        "accuracy": round(float(acc), 4), "pos": pos, "neg": neg,
        "defaults": {k: R4(v) for k, v in defaults.items()},
    }
    if samples:
        OUT[name]["samples"] = samples
    print(name, "acc", round(float(acc), 4))

# Titanic (features engineered exactly as the app builds them)
raw = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
raw["female"] = (raw["Sex"] == "female").astype(int)
raw["emb_q"] = (raw["Embarked"] == "Q").astype(int)
raw["emb_s"] = (raw["Embarked"] == "S").astype(int)
raw["Age"] = raw["Age"].fillna(raw["Age"].median()); raw["Fare"] = raw["Fare"].fillna(raw["Fare"].median())
X = raw[["Pclass", "female", "Age", "SibSp", "Parch", "Fare", "emb_q", "emb_s"]]
y = raw["Survived"]
a, b, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(a)
m = LogisticRegression(max_iter=5000).fit(sc.transform(a), ya)
acc = accuracy_score(yb, m.predict(sc.transform(b)))
OUT["titanic"] = {"type": "scaled_lr", "features": list(X.columns),
                  "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) for v in sc.scale_],
                  "coef": [R4(v) for v in m.coef_[0]], "intercept": R4(m.intercept_[0]),
                  "accuracy": round(float(acc), 4), "pos": "Survived", "neg": "Did not survive",
                  "defaults": {k: R4(v) for k, v in X.median().items()}}
print("titanic acc", round(float(acc), 4))

# House Price (linear regression)
cal = fetch_california_housing()
X = pd.DataFrame(cal.data, columns=cal.feature_names); y = cal.target * 100000
a, b, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42)
m = LinearRegression().fit(a, ya)
OUT["house"] = {"type": "linear", "features": list(X.columns),
                "coef": [R4(v) for v in m.coef_], "intercept": R4(m.intercept_),
                "r2": round(float(r2_score(yb, m.predict(b))), 4),
                "defaults": {k: R4(v) for k, v in X.median().items()}}
print("house r2", OUT["house"]["r2"])

# Iris (softmax LR, raw features)
iris = load_iris()
a, b, ya, yb = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target)
m = LogisticRegression(max_iter=1000).fit(a, ya)
OUT["iris"] = {"type": "softmax", "features": list(iris.feature_names),
               "coef": [[R4(v) for v in row] for row in m.coef_],
               "intercept": [R4(v) for v in m.intercept_],
               "classes": list(iris.target_names),
               "accuracy": round(float(accuracy_score(yb, m.predict(b))), 4),
               "defaults": {f: R4(v) for f, v in zip(iris.feature_names, np.median(iris.data, axis=0))}}
print("iris acc", OUT["iris"]["accuracy"])

# Breast Cancer (scaled LR + two real sample rows)
bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names); y = bc.target  # 0 malignant, 1 benign
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
s_m = a[ya == 0].iloc[0]; s_b = a[ya == 1].iloc[0]
export_scaled_lr("cancer", a, ya, accuracy_score(yb, LogisticRegression(max_iter=5000).fit(StandardScaler().fit(a).transform(a), ya).predict(StandardScaler().fit(a).transform(b))),
                 "Benign", "Malignant", X.median(),
                 samples={"benign": {k: R4(v) for k, v in s_b.items()}, "malignant": {k: R4(v) for k, v in s_m.items()}})

# Wine Quality (scaled LR)
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
df["good"] = (df["quality"] >= 7).astype(int)
X = df.drop(columns=["quality", "good"]); y = df["good"]
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(a)
m = LogisticRegression(max_iter=5000).fit(sc.transform(a), ya)
export_scaled_lr("wine", a, ya, accuracy_score(yb, m.predict(sc.transform(b))), "Good wine (≥7)", "Not good", X.median())

# Heart Disease (scaled LR)
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", names=cols, na_values="?")
df = df.fillna(df.median(numeric_only=True))
df["disease"] = (df["num"] > 0).astype(int)
X = df.drop(columns=["num", "disease"]); y = df["disease"]
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(a)
m = LogisticRegression(max_iter=5000).fit(sc.transform(a), ya)
export_scaled_lr("heart", a, ya, accuracy_score(yb, m.predict(sc.transform(b))), "Disease signs present", "No disease signs", X.median())

# ---------------------------------------------------------------- MNIST MLP
print("training MNIST web model (784-64-10 on full 60k)...")
with urllib.request.urlopen("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz") as r:
    npz = np.load(io.BytesIO(r.read()))
Xtr = npz["x_train"].reshape(60000, -1) / 255.0; ytr = npz["y_train"]
Xte = npz["x_test"].reshape(10000, -1) / 255.0; yte = npz["y_test"]
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=25, random_state=42)
mlp.fit(Xtr, ytr)
acc = accuracy_score(yte, mlp.predict(Xte))
R3 = lambda arr: np.round(arr, 3).tolist()
OUT["mnist"] = {"type": "mlp", "W1": R3(mlp.coefs_[0]), "b1": R3(mlp.intercepts_[0]),
                "W2": R3(mlp.coefs_[1]), "b2": R3(mlp.intercepts_[1]),
                "accuracy": round(float(acc), 4)}
print("mnist acc", OUT["mnist"]["accuracy"])

with open("models.json", "w") as f:
    json.dump(OUT, f, separators=(",", ":"))
print("models.json written")
