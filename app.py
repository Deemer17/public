import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ============================================================
# NLTK PREP
# ============================================================
def load_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

load_nltk()
stopwords_ind = stopwords.words("indonesian")


# ============================================================
# CLEANING FUNCTIONS
# ============================================================
def is_noise(a):
    a = str(a).lower().strip()
    if re.fullmatch(r"[0-9\-\+\(\) ]{7,}", a):
        return True
    bad = ["pt", "cv", "tbk", "bank", "bca", "kantor", "industri", "factory"]
    return any(b in a for b in bad)

def bersih(t):
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ============================================================
# KECAMATAN LIST
# ============================================================
KEC_LIST = [
    "Bekasi Timur", "Bekasi Selatan", "Bekasi Utara", "Bekasi Barat",
    "Jatiasih", "Jatisampurna", "Rawalumbu", "Mustika Jaya",
    "Pondokgede", "Pondok Melati", "Medan Satria", "Bantargebang"
]

MAPPING = {
    "Jatikramat": "Jatiasih",
    "Jati Kramat": "Jatiasih",
    "Jaticempaka": "Pondokgede",
    "Galaxy": "Bekasi Selatan",
    "Aren Jaya": "Bekasi Timur",
    "Pejuang": "Medan Satria",
}

def detect_kec(a):
    a = str(a)
    for k, v in MAPPING.items():
        if k.lower() in a.lower():
            return v
    for k in KEC_LIST:
        if k.lower() in a.lower():
            return k
    return None


# ============================================================
# GEOJSON UTIL
# ============================================================
def get_polygon_from_geojson(geo, kecamatan_up):
    for feat in geo["features"]:
        if feat["properties"]["KECAMATAN"].strip().upper() == kecamatan_up.strip().upper():
            return feat["geometry"]["coordinates"][0]
    return None

def get_centroid_from_geojson(geo, kec):
    for feat in geo["features"]:
        if feat["properties"]["KECAMATAN"].title() == kec.title():
            coords = np.array(feat["geometry"]["coordinates"][0])
            lon0 = coords[:,0].mean()
            lat0 = coords[:,1].mean()
            return lat0, lon0
    return None, None


# ============================================================
# CACHED PROCESSING PIPELINE
# ============================================================
@st.cache_data(show_spinner=True)
def process_data(uploaded_excel, uploaded_geojson, n_text_clusters=5, n_kec_clusters=4):

    df = pd.read_excel(uploaded_excel)

    if "ALAMAT" not in df.columns or "PERMASALAHAN" not in df.columns:
        raise ValueError("File Excel wajib memiliki kolom ALAMAT dan PERMASALAHAN")

    # cleaning
    df = df[~df["ALAMAT"].astype(str).apply(is_noise)]
    df["ALAMAT"] = df["ALAMAT"].astype(str).str.title().str.strip()
    df["KECAMATAN"] = df["ALAMAT"].apply(detect_kec)
    df = df.dropna(subset=["KECAMATAN"])
    df["KECAMATAN_UP"] = df["KECAMATAN"].str.upper()

    # text clean + tfidf
    df["PERMASALAHAN_CLEAN"] = df["PERMASALAHAN"].apply(bersih)
    tfidf = TfidfVectorizer(stop_words=stopwords_ind, max_features=2000)
    X_text = tfidf.fit_transform(df["PERMASALAHAN_CLEAN"])

    # cluster keluhan
    km_text = KMeans(n_clusters=n_text_clusters, random_state=42, n_init=10)
    df["CLUSTER_KELUHAN"] = km_text.fit_predict(X_text)

    # silhouette
    sil_score = silhouette_score(X_text, df["CLUSTER_KELUHAN"])

    # tfidf terms per cluster
    terms = tfidf.get_feature_names_out()
    term_top = {}
    for i in range(n_text_clusters):
        center = km_text.cluster_centers_[i]
        top = center.argsort()[-15:][::-1]
        term_top[i] = [terms[t] for t in top]

    # aggregate per kecamatan
    kec_total = df.groupby("KECAMATAN").size().reset_index(name="TOTAL_KELUHAN")
    kec_comp = df.pivot_table(
        index="KECAMATAN",
        columns="CLUSTER_KELUHAN",
        aggfunc="size",
        fill_value=0
    )
    kec_comp.columns = [f"CL_{c}" for c in kec_comp.columns]
    kec_features = kec_total.merge(kec_comp, on="KECAMATAN")
    kec_features["KECAMATAN_UP"] = kec_features["KECAMATAN"].str.upper()

    # cluster kecamatan
    scaler = StandardScaler()
    X_kec = scaler.fit_transform(kec_features.drop(columns=["KECAMATAN","KECAMATAN_UP"]))
    km_kec = KMeans(n_clusters=n_kec_clusters, random_state=42, n_init=10)
    kec_features["CLUSTER_KECAMATAN"] = km_kec.fit_predict(X_kec)

    # load geojson
    geo = json.load(uploaded_geojson)

    return df, kec_features, geo, sil_score, term_top


# ============================================================
# FIXED JITTER (NO FLICKERING)
# ============================================================
def jitter_point(lat, lon, index):
    np.random.seed(index)
    lat_j = lat + np.random.uniform(-0.003, 0.003)
    lon_j = lon + np.random.uniform(-0.003, 0.003)
    return lat_j, lon_j


# ============================================================
# CACHED POINTS FOR DOT & HEATMAP
# ============================================================
@st.cache_data
def get_dot_points(df_cl, geo):
    pts = []
    for i, row in df_cl.iterrows():
        lat0, lon0 = get_centroid_from_geojson(geo, row["KECAMATAN"])
        if lat0 is None:
            continue
        pts.append(jitter_point(lat0, lon0, i))
    return pts

@st.cache_data
def get_heatmap_points(df_cl, geo):
    pts = []
    for _, row in df_cl.iterrows():
        lat0, lon0 = get_centroid_from_geojson(geo, row["KECAMATAN"])
        if lat0 is None:
            continue
        pts.append([lat0, lon0])
    return pts


# ============================================================
# MAP BUILDERS
# ============================================================
def map_dot(df, geo, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    points = get_dot_points(df_cl, geo)

    m = folium.Map(location=[-6.25,107.03], zoom_start=12)
    color = ["#E53935","#FB8C00","#FDD835","#43A047","#1E88E5"][cluster_id]

    for lat, lon in points:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(m)
    return m


def map_heat(df, geo, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    points = get_heatmap_points(df_cl, geo)

    m = folium.Map(location=[-6.25,107.03], zoom_start=12)
    if points:
        HeatMap(points, radius=25, blur=30).add_to(m)
    return m


def map_polygon(df, kec_features, geo, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    group = df_cl.groupby("KECAMATAN").size().reset_index(name="JML")

    kec_cl = kec_features.merge(group, on="KECAMATAN", how="left")
    kec_cl["JML"] = kec_cl["JML"].fillna(0)

    m = folium.Map(location=[-6.25,107.03], zoom_start=12)

    def color(v):
        if v == 0: return "#E0E0E0"
        if v <= 5: return "#FFF176"
        if v <= 15: return "#FFB74D"
        return "#E53935"

    for _, row in kec_cl.iterrows():
        coords = get_polygon_from_geojson(geo, row["KECAMATAN_UP"])
        if coords is None:
            continue
        folium.Polygon(
            locations=[(lat,lon) for lon,lat in coords],
            fill=True,
            fill_color=color(row["JML"]),
            color="black",
            weight=1,
            tooltip=f"{row['KECAMATAN']} – {int(row['JML'])} kasus"
        ).add_to(m)
    return m


# ============================================================
# VISUAL – WORDCLOUD & BAR CHART
# ============================================================
def show_wordcloud(df, cluster_id):
    text = " ".join(df[df["CLUSTER_KELUHAN"]==cluster_id]["PERMASALAHAN_CLEAN"])
    wc = WordCloud(width=800,height=400, background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def show_bar(df, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    count = df_cl["KECAMATAN"].value_counts()

    fig, ax = plt.subplots(figsize=(6,4))
    count.plot(kind="bar", ax=ax)
    ax.set_title(f"Distribusi Keluhan per Kecamatan – Cluster {cluster_id}")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


# ============================================================
# STREAMLIT MAIN APP
# ============================================================
def main():
    st.title("📍 Pemetaan Keluhan Kota Bekasi")
    st.write("Visualisasi Heatmap, Poligon, Dot Density, Wordcloud, dan Analisis TF-IDF")

    with st.sidebar:
        st.header("Upload Data")
        excel = st.file_uploader("Upload Excel (contoh: 2024.xlsx)", type=["xlsx"])
        geo = st.file_uploader("Upload GeoJSON (bekasi_kecamatan.geojson)", type=["geojson","json"])

        st.header("Parameter Clustering")
        n_text_clusters = st.slider("Cluster Keluhan", 2, 8, 5)
        n_kec_clusters = st.slider("Cluster Kecamatan", 2, 8, 4)

    if not excel or not geo:
        st.info("Silakan upload file Excel dan GeoJSON.")
        return

    df, kec_features, geojson, sil, terms = process_data(
        excel, geo, n_text_clusters, n_kec_clusters
    )

    # summary
    st.subheader("Ringkasan Analisis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Keluhan", len(df))
    c2.metric("Jumlah Kecamatan", df["KECAMATAN"].nunique())
    c3.metric("Silhouette", f"{sil:.3f}")

    # silhouette bar
    fig, ax = plt.subplots(figsize=(4,1.5))
    ax.barh(["Silhouette"], [sil], color="green")
    ax.set_xlim(0,1)
    st.pyplot(fig)

    # TF-IDF terms
    st.subheader("Top TF-IDF Terms per Cluster")
    pilih = st.selectbox("Pilih cluster:", sorted(terms.keys()))
    st.write(", ".join(terms[pilih]))

    # maps
    st.subheader("Visualisasi Peta per Cluster")
    cl = st.number_input("Cluster ID", 0, n_text_clusters-1, 0)
    tipe = st.radio("Jenis Peta", ["Peta Poligon","Heatmap","Dot Density"], horizontal=True)

    if tipe == "Peta Poligon":
        map_obj = map_polygon(df, kec_features, geojson, cl)
    elif tipe == "Heatmap":
        map_obj = map_heat(df, geojson, cl)
    else:
        map_obj = map_dot(df, geojson, cl)

    st_folium(map_obj, width=700, height=500)

    # wordcloud & bar
    st.subheader("Wordcloud & Distribusi Keluhan")
    col_a, col_b = st.columns(2)
    with col_a:
        show_wordcloud(df, cl)
    with col_b:
        show_bar(df, cl)


if __name__ == "__main__":
    main()
