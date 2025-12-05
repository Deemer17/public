import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords

# =====================================================================
# 1. Title
# =====================================================================
st.title("📊 Analisis Keluhan Masyarakat Berbasis Clustering – Kota Bekasi")
st.write("Aplikasi ini memproses file Excel berisi data keluhan, melakukan clustering, dan menampilkan peta interaktif.")

# =====================================================================
# 2. Upload File
# =====================================================================
uploaded_file = st.file_uploader("📁 Upload file Excel (format .xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.warning("Silakan upload file terlebih dahulu.")
    st.stop()

# Load data
df = pd.read_excel(uploaded_file)

# =====================================================================
# 3. Cleaning Data
# =====================================================================
def is_noise(a):
    a = str(a).lower().strip()
    if re.fullmatch(r"[0-9\-\+\(\) ]{7,}", a):
        return True
    bad = ["pt","cv","tbk","bank","bca","kantor","industri","factory"]
    return any(b in a for b in bad)

df = df[~df["ALAMAT"].astype(str).apply(is_noise)].copy()
df["ALAMAT"] = df["ALAMAT"].astype(str).str.title().str.strip()

# Mapping Kecamatan
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

# =====================================================================
# 4. Clustering Text Keluhan
# =====================================================================
stopwords_ind = stopwords.words("indonesian")

def bersih(t):
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    return re.sub(r"\s+"," ",t).strip()

df["PERMASALAHAN_CLEAN"] = df["PERMASALAHAN"].apply(bersih)

tfidf = TfidfVectorizer(stop_words=stopwords_ind, max_features=1500)
X_text = tfidf.fit_transform(df["PERMASALAHAN_CLEAN"])

kmeans_keluhan = KMeans(n_clusters=5, random_state=42)
df["CLUSTER_KELUHAN"] = kmeans_keluhan.fit_predict(X_text)

# =====================================================================
# 5. Feature Kecamatan
# =====================================================================
kec_total = df.groupby("KECAMATAN").size().reset_index(name="TOTAL_KELUHAN")
kec_comp = df.pivot_table(index="KECAMATAN", columns="CLUSTER_KELUHAN", aggfunc="size", fill_value=0)
kec_comp.columns = [f"CL_{c}" for c in kec_comp.columns]
kec_features = kec_total.merge(kec_comp, on="KECAMATAN")
num_cols = [c for c in kec_features.columns if c != "KECAMATAN"]

scaler = StandardScaler()
X_kec = scaler.fit_transform(kec_features[num_cols])

kmeans_kec = KMeans(n_clusters=4, random_state=42)
kec_features["CLUSTER_KECAMATAN"] = kmeans_kec.fit_predict(X_kec)
kec_features["KECAMATAN_UP"] = kec_features["KECAMATAN"].str.upper()

# =====================================================================
# 6. Load GeoJSON Bekasi
# =====================================================================
try:
    geo = json.load(open("bekasi_kecamatan.geojson"))
except:
    st.error("❌ File `bekasi_kecamatan.geojson` wajib di-upload di folder yang sama.")
    st.stop()

# =====================================================================
# 7. Fungsi Peta Polygon Keluhan
# =====================================================================
cluster_colors = ["#E53935","#FB8C00","#FDD835","#43A047","#1E88E5"]

def show_polygon(cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    kec_cl = df_cl.groupby("KECAMATAN").size().reset_index(name="JUMLAH_CLUSTER")
    kec_cl = kec_features.merge(kec_cl, on="KECAMATAN", how="left")
    kec_cl["JUMLAH_CLUSTER"] = kec_cl["JUMLAH_CLUSTER"].fillna(0)

    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)

    for _, row in kec_cl.iterrows():
        kec = row["KECAMATAN_UP"]
        jumlah = row["JUMLAH_CLUSTER"]

        coords = None
        for feat in geo["features"]:
            if feat["properties"]["KECAMATAN"].upper() == kec.upper():
                coords = feat["geometry"]["coordinates"][0]
                break
        if coords is None:
            continue

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            fill=True,
            fill_color=cluster_colors[cluster_id],
            fill_opacity=0.5,
            color="black",
            weight=1,
            tooltip=f"{kec} | Keluhan CL{cluster_id}: {int(jumlah)}"
        ).add_to(m)

    return m

# =====================================================================
# 8. Wordcloud
# =====================================================================
def wordcloud_cluster(cluster_id):
    text = " ".join(df[df["CLUSTER_KELUHAN"] == cluster_id]["PERMASALAHAN_CLEAN"])
    wc = WordCloud(width=900, height=400, background_color="white").generate(text)
    fig = plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig

# =====================================================================
# 9. INTERFACE STREAMLIT
# =====================================================================
st.header("🗂 Pilih Cluster Keluhan")
cluster_sel = st.selectbox("Pilih Cluster", [0,1,2,3,4])

st.subheader(f"📌 Peta Poligon Cluster {cluster_sel}")
peta = show_polygon(cluster_sel)
st_folium(peta, width=700, height=450)

st.subheader(f"☁ Wordcloud Cluster {cluster_sel}")
fig = wordcloud_cluster(cluster_sel)
st.pyplot(fig)

st.success("Analisis selesai. Anda bisa mengganti cluster untuk melihat perbedaan.")

