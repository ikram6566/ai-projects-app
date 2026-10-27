#!/usr/bin/env python3
"""Batch-7 web exports: airfoil/energy/pm25/car as linear models, dry bean as tabular softmax."""
import io, json, os, urllib.request, zipfile
import numpy as np
import pandas as pd

OUT = json.load(open("models.json"))
R4 = lambda x: round(float(x), 4)
def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

os.makedirs("models", exist_ok=True)
tab = {}

def linear_export(key, X, y, note_r2=None):
    a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42)
    m = LinearRegression().fit(a, ya)
    pv = m.predict(b)
    tab[key] = {"type": "linear", "features": list(X.columns),
                "coef": [R4(v) for v in m.coef_], "intercept": R4(m.intercept_),
                "r2": round(float(r2_score(yb, pv)), 4), "mae": round(float(mean_absolute_error(yb, pv)), 2),
                "defaults": {k: R4(v) for k, v in X.median().items()}}
    print(key, "linear r2", tab[key]["r2"], "mae", tab[key]["mae"], "features", X.shape[1])

# airfoil
raw = get("https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".dat")][0]
cols = ["frequency", "angle_of_attack", "chord_length", "free_stream_velocity", "suction_side_thickness", "scaled_sound_pressure"]
d = pd.read_csv(io.BytesIO(zf.read(name)), sep=r"\s+", header=None, names=cols)
linear_export("airfoil", d.drop(columns=["scaled_sound_pressure"]), d["scaled_sound_pressure"])

# energy (heating load)
raw = get("https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
d = pd.read_excel(io.BytesIO(zf.read(name)), engine="calamine")
d.columns = ["relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height",
             "orientation", "glazing_area", "glazing_distribution", "heating_load", "cooling_load"]
linear_export("energy", d.drop(columns=["heating_load", "cooling_load"]), d["heating_load"])

# pm25 (no wind-direction dummies in the browser version)
raw = get("https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".csv")][0]
d = pd.read_csv(io.BytesIO(zf.read(name))).dropna(subset=["pm2.5"])
X = d[["month", "day", "hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]]
linear_export("pm25", X, d["pm2.5"])

# car (numeric features only in the browser version)
raw = get("https://archive.ics.uci.edu/static/public/10/automobile.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".data")][0]
cols = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration", "num_doors", "body_style", "drive_wheels",
        "engine_location", "wheel_base", "length", "width", "height", "curb_weight", "engine_type",
        "num_cylinders", "engine_size", "fuel_system", "bore", "stroke", "compression_ratio", "horsepower",
        "peak_rpm", "city_mpg", "highway_mpg", "price"]
d = pd.read_csv(io.BytesIO(zf.read(name)), header=None, names=cols, na_values="?").dropna(subset=["price"])
num = ["wheel_base", "length", "width", "height", "curb_weight", "engine_size", "bore", "stroke",
       "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg"]
X = d[num].apply(pd.to_numeric, errors="coerce")
ok = X.notna().all(axis=1)
linear_export("car", X[ok], d["price"][ok])

json.dump(tab, open("models/m_tabular5.json", "w"), separators=(",", ":"))
print("m_tabular5.json", os.path.getsize("models/m_tabular5.json") // 1024, "KB")

# dry bean — tabular softmax (scaled multinomial LR)
raw = get("https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip")
zf = zipfile.ZipFile(io.BytesIO(raw))
name = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
d = pd.read_excel(io.BytesIO(zf.read(name)), engine="calamine")
X, y = d.drop(columns=["Class"]), d["Class"]
a, b, ya, yb = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
sc = StandardScaler().fit(a)
m = LogisticRegression(max_iter=1000).fit(sc.transform(a), ya)
acc = accuracy_score(yb, m.predict(sc.transform(b)))
OUT["bean"] = {"type": "softmax_tab", "classes": list(m.classes_), "features": list(X.columns),
               "mean": [R4(v) for v in sc.mean_], "scale": [R4(v) if v != 0 else 1.0 for v in sc.scale_],
               "coef": [[R4(v) for v in row] for row in m.coef_],
               "intercept": [R4(v) for v in m.intercept_],
               "accuracy": round(float(acc), 4),
               "defaults": {k: R4(v) for k, v in X.median().items()}}
json.dump({"bean": OUT["bean"]}, open("models/m_bean.json", "w"), separators=(",", ":"))
print("bean softmax acc", OUT["bean"]["accuracy"], "file", os.path.getsize("models/m_bean.json") // 1024, "KB")

OUT.update(tab)
with open("models.json", "w") as f:
    json.dump(OUT, f, separators=(",", ":"))
print("models.json updated")
