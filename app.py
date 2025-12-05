!pip install folium
import pandas as pd
import numpy as np
import json
import re
import folium
import requests

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
from nltk.corpus import stopwords

from google.colab import files
print("Upload file 2024.xlsx")
uploaded = files.upload()

df = pd.read_excel(next(iter(uploaded)))
df.head()

def is_noise(a):
    a = str(a).lower().strip()
    if re.fullmatch(r"[0-9\-\+\(\) ]{7,}", a):
        return True
    bad = ["pt","cv","tbk","bank","bca","kantor","industri","factory"]
    return any(b in a for b in bad)

df = df[~df["ALAMAT"].astype(str).apply(is_noise)].copy()
df["ALAMAT"] = df["ALAMAT"].astype(str).str.title().str.strip()

print("Setelah cleaning:", len(df))

kec_list = [
    "Bekasi Timur","Bekasi Selatan","Bekasi Utara","Bekasi Barat",
    "Jatiasih","Jatisampurna","Rawalumbu","Mustika Jaya",
    "Pondokgede","Pondok Melati","Medan Satria","Bantargebang"
]

mapping = {
    "Jatikramat": "Jatiasih",
    "Jati Kramat": "Jatiasih",
    "Jaticempaka": "Pondokgede",
    "Galaxy": "Bekasi Selatan",
    "Aren Jaya": "Bekasi Timur",
    "Pejuang": "Medan Satria"
}

df["KECAMATAN"] = df["ALAMAT"].map(mapping)

def detect_kec(a):
    for k in kec_list:
        if k.lower() in a.lower():
            return k
    return None

df["KECAMATAN"] = df["KECAMATAN"].fillna(df["ALAMAT"].apply(detect_kec))
df = df.dropna(subset=["KECAMATAN"])
df.head()

kec_list = [
    "Bekasi Timur","Bekasi Selatan","Bekasi Utara","Bekasi Barat",
    "Jatiasih","Jatisampurna","Rawalumbu","Mustika Jaya",
    "Pondokgede","Pondok Melati","Medan Satria","Bantargebang"
]

mapping = {
    "Jatikramat": "Jatiasih",
    "Jati Kramat": "Jatiasih",
    "Jaticempaka": "Pondokgede",
    "Galaxy": "Bekasi Selatan",
    "Aren Jaya": "Bekasi Timur",
    "Pejuang": "Medan Satria"
}

df["KECAMATAN"] = df["ALAMAT"].map(mapping)

def detect_kec(a):
    for k in kec_list:
        if k.lower() in a.lower():
            return k
    return None

df["KECAMATAN"] = df["KECAMATAN"].fillna(df["ALAMAT"].apply(detect_kec))
df = df.dropna(subset=["KECAMATAN"])
df.head()

stopwords_ind = stopwords.words("indonesian")

def bersih(t):
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+"," ",t).strip()
    return t

df["PERMASALAHAN_CLEAN"] = df["PERMASALAHAN"].apply(bersih)

tfidf = TfidfVectorizer(stop_words=stopwords_ind, max_features=2000)
X_text = tfidf.fit_transform(df["PERMASALAHAN_CLEAN"])

kmeans_keluhan = KMeans(n_clusters=5, random_state=42)
df["CLUSTER_KELUHAN"] = kmeans_keluhan.fit_predict(X_text)

df[["PERMASALAHAN","CLUSTER_KELUHAN"]].head(1000)

kec_total = df.groupby("KECAMATAN").size().reset_index(name="TOTAL_KELUHAN")

kec_comp = df.pivot_table(
    index="KECAMATAN",
    columns="CLUSTER_KELUHAN",
    aggfunc="size",
    fill_value=0
)
kec_comp.columns = [f"CL_{c}" for c in kec_comp.columns]

kec_features = kec_total.merge(kec_comp, on="KECAMATAN")
kec_features

num_cols = [c for c in kec_features.columns if c != "KECAMATAN"]

scaler = StandardScaler()
X_kec = scaler.fit_transform(kec_features[num_cols])

kmeans_kec = KMeans(n_clusters=4, random_state=42)
kec_features["CLUSTER_KECAMATAN"] = kmeans_kec.fit_predict(X_kec)

kec_features["KECAMATAN_UP"] = kec_features["KECAMATAN"].str.upper()
kec_features

from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_text, df["CLUSTER_KELUHAN"])
print("Silhouette Score:", sil_score)

terms = tfidf.get_feature_names_out()
for i in range(5):
    print(f"Cluster {i}:")
    center = kmeans_keluhan.cluster_centers_[i]
    top = center.argsort()[-15:][::-1]
    print([terms[t] for t in top])

import json

with open("bekasi_kecamatan.geojson", "r") as f:
    geo = json.load(f)

map_cl = folium.Map(location=[-6.25, 107.03], zoom_start=12)

colors = ["red","orange","green","blue","purple"]

for _, row in kec_features.iterrows():
    kec = row["KECAMATAN_UP"]
    cluster = row["CLUSTER_KECAMATAN"]

    coords = None  # default kosong

    # cari kecamatan di geojson
    for feat in geo["features"]:
        nama_geo = feat["properties"]["KECAMATAN"].strip().upper()
        nama_df  = kec.strip().upper()

        if nama_geo == nama_df:
            coords = feat["geometry"]["coordinates"][0]
            break

    # kalau coords tetap None → ada mismatch penamaan
    if coords is None:
        print(f"⚠ WARNING: Kecamatan '{kec}' tidak ditemukan di geojson.")
        continue  # skip kecamatan tanpa polygon

    # buat polygon
    folium.Polygon(
        locations=[(lat, lon) for lon, lat in coords],
        fill=True,
        fill_color=colors[cluster],
        color="black",
        weight=1,
        tooltip=f"{kec} — Cluster {cluster}"
    ).add_to(map_cl)

map_cl

# ==========================================================
# FUNGSI PEMBUAT PETA PER CLUSTER KELUHAN
# ==========================================================

import folium

def peta_cluster(cluster_id):
    print(f"Menampilkan Peta Cluster {cluster_id} ...")

    # Filter data berdasarkan cluster
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]

    # Hitung jumlah keluhan cluster ini per kecamatan
    cl_kec = df_cl.groupby("KECAMATAN").size().reset_index(name="JUMLAH_CLUSTER")

    # Gabungkan ke kec_features agar struktur sama
    kec_cl = kec_features.merge(cl_kec, on="KECAMATAN", how="left")
    kec_cl["JUMLAH_CLUSTER"] = kec_cl["JUMLAH_CLUSTER"].fillna(0)

    # Buat peta dasar
    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)

    # Warna heat berdasarkan jumlah cluster
    def get_color(v):
        if v == 0:
            return "#E0E0E0"        # abu-abu
        elif v <= 5:
            return "#FFF176"        # kuning
        elif v <= 15:
            return "#FFB74D"        # oranye
        else:
            return "#E53935"        # merah

    # Loop tiap kecamatan → gambar wilayah
    for _, row in kec_cl.iterrows():
        kec = row["KECAMATAN_UP"]
        jumlah = row["JUMLAH_CLUSTER"]

        coords = None
        for feat in geo["features"]:
            if feat["properties"]["KECAMATAN"].strip().upper() == kec.strip().upper():
                coords = feat["geometry"]["coordinates"][0]
                break

        if coords is None:
            print(f"⚠ Tidak ditemukan di GeoJSON: {kec}")
            continue

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            fill=True,
            fill_color=get_color(jumlah),
            fill_opacity=0.8,
            color="black",
            weight=1,
            tooltip=f"{kec}<br>Keluhan CL{cluster_id}: {int(jumlah)}"
        ).add_to(m)

    return m

peta_cluster(0)
peta_cluster(1)
peta_cluster(2)
peta_cluster(3)
peta_cluster(4)

import numpy as np
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def get_total(kec):
    nama = kec.strip().upper()
    match = kec_features.loc[kec_features["KECAMATAN_UP"] == nama, "TOTAL_KELUHAN"]
    return int(match.values[0]) if len(match) > 0 else 0

def peta_cluster(cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    cl_kec = df_cl.groupby("KECAMATAN").size().reset_index(name="JUMLAH_CLUSTER")
    kec_cl = kec_features.merge(cl_kec, on="KECAMATAN", how="left")
    kec_cl["JUMLAH_CLUSTER"] = kec_cl["JUMLAH_CLUSTER"].fillna(0)

    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)

    def get_color(v):
        if v == 0:
            return "#E0E0E0"
        elif v <= 5:
            return "#FFF176"
        elif v <= 15:
            return "#FFB74D"
        else:
            return "#E53935"

    for _, row in kec_cl.iterrows():
        kec = row["KECAMATAN_UP"]
        jumlah = row["JUMLAH_CLUSTER"]

        coords = None
        for feat in geo["features"]:
            if feat["properties"]["KECAMATAN"].strip().upper() == kec.strip().upper():
                coords = feat["geometry"]["coordinates"][0]
                break

        if coords is None:
            print(f"⚠ Tidak ditemukan di GeoJSON: {kec}")
            continue

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            fill=True,
            fill_color=get_color(jumlah),
            fill_opacity=0.8,
            color="black",
            weight=1,
            tooltip=f"{kec}<br>Keluhan CL{cluster_id}: {int(jumlah)}"
        ).add_to(m)

    return m

def heatmap_cluster(cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)
    heat_points = []

    for _, row in df_cl.iterrows():
        kec = row["KECAMATAN"]

        lat0, lon0 = None, None
        for feat in geo["features"]:
            if feat["properties"]["KECAMATAN"].title() == kec.title():
                coords = np.array(feat["geometry"]["coordinates"][0])
                lon0 = coords[:,0].mean()
                lat0 = coords[:,1].mean()
                break

        if lat0 is None:
            continue

        heat_points.append([
            lat0 + np.random.uniform(-0.005, 0.005),
            lon0 + np.random.uniform(-0.005, 0.005),
        ])

    if heat_points:
        HeatMap(heat_points, radius=25, blur=30).add_to(m)

    return m

cluster_colors = {
    0: "#E53935",  # merah
    1: "#FB8C00",  # oranye
    2: "#FDD835",  # kuning
    3: "#43A047",  # hijau
    4: "#1E88E5",  # biru
}

def dot_cluster(cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)

    for _, row in df_cl.iterrows():
        kec = row["KECAMATAN"]

        lat0, lon0 = None, None
        for feat in geo["features"]:
            if feat["properties"]["KECAMATAN"].title() == kec.title():
                coords = np.array(feat["geometry"]["coordinates"][0])
                lon0 = coords[:,0].mean()
                lat0 = coords[:,1].mean()
                break

        if lat0 is None:
            continue

        folium.CircleMarker(
            location=[
                lat0 + np.random.uniform(-0.004, 0.004),
                lon0 + np.random.uniform(-0.004, 0.004),
            ],
            radius=3,
            color=cluster_colors[cluster_id],
            fill=True,
            fill_color=cluster_colors[cluster_id],
            fill_opacity=0.8,
        ).add_to(m)

    return m

def wordcloud_cluster(cluster_id):
    text = " ".join(df[df["CLUSTER_KELUHAN"] == cluster_id]["PERMASALAHAN_CLEAN"])
    if not text.strip():
        print(f"Tidak ada teks untuk cluster {cluster_id}")
        return

    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="viridis"
    ).generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Wordcloud Cluster {cluster_id}")
    plt.show()

def bar_cluster(cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    kec_count = df_cl["KECAMATAN"].value_counts()

    plt.figure(figsize=(10,5))
    kec_count.plot(kind="bar")
    plt.title(f"Jumlah Keluhan per Kecamatan – Cluster {cluster_id}")
    plt.xlabel("Kecamatan")
    plt.ylabel("Jumlah Keluhan")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

for c in range(5):
    print("\n====================================")
    print(f"              CLUSTER {c}           ")
    print("====================================\n")

    # Peta poligon
    display(peta_cluster(c))

    # Heatmap
    display(heatmap_cluster(c))

    # Dot density
    display(dot_cluster(c))

    # Wordcloud
    wordcloud_cluster(c)

    # Bar chart
    bar_cluster(c)
