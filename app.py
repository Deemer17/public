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

# =====================================================================
# SETUP NLTK
# =====================================================================
# Download resource kalau belum ada
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stopwords_ind = stopwords.words("indonesian")

# =====================================================================
# FUNGSI UTIL
# =====================================================================

def is_noise(a):
    a = str(a).lower().strip()
    # nomor telepon
    if re.fullmatch(r"[0-9\-\+\(\) ]{7,}", a):
        return True
    bad = ["pt", "cv", "tbk", "bank", "bca", "kantor", "industri", "factory"]
    return any(b in a for b in bad)

def bersih(t):
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# list kecamatan & mapping manual (sesuai kode asli)
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

def detect_kec(a: str):
    a = str(a)
    # cek mapping manual dulu
    for key, val in MAPPING.items():
        if key.lower() in a.lower():
            return val
    # cek nama kecamatan langsung
    for k in KEC_LIST:
        if k.lower() in a.lower():
            return k
    return None

# cari koordinat polygon kecamatan di geojson
def get_polygon_from_geojson(geo, kecamatan_up):
    for feat in geo["features"]:
        nama_geo = feat["properties"]["KECAMATAN"].strip().upper()
        if nama_geo == kecamatan_up.strip().upper():
            # asumsi polygon sederhana
            return feat["geometry"]["coordinates"][0]
    return None

def get_centroid_from_geojson(geo, kecamatan_title):
    for feat in geo["features"]:
        if feat["properties"]["KECAMATAN"].title() == kecamatan_title.title():
            coords = np.array(feat["geometry"]["coordinates"][0])
            lon0 = coords[:, 0].mean()
            lat0 = coords[:, 1].mean()
            return lat0, lon0
    return None, None

# =====================================================================
# LOGIKA UTAMA: PREPROCESS + CLUSTERING
# =====================================================================

@st.cache_data(show_spinner=True)
def process_data(uploaded_excel, uploaded_geojson, n_text_clusters=5, n_kec_clusters=4):

    # -------------------------
    # 1. Baca data & cleaning
    # -------------------------
    df = pd.read_excel(uploaded_excel)

    # pastikan kolom wajib
    if "ALAMAT" not in df.columns or "PERMASALAHAN" not in df.columns:
        raise ValueError("File Excel harus punya kolom 'ALAMAT' dan 'PERMASALAHAN'.")

    # buang alamat noise
    df = df[~df["ALAMAT"].astype(str).apply(is_noise)].copy()
    df["ALAMAT"] = df["ALAMAT"].astype(str).str.title().str.strip()

    # deteksi kecamatan
    df["KECAMATAN"] = df["ALAMAT"].apply(detect_kec)
    df = df.dropna(subset=["KECAMATAN"])
    df["KECAMATAN_UP"] = df["KECAMATAN"].str.upper()

    # -------------------------
    # 2. Text cleaning & TF-IDF
    # -------------------------
    df["PERMASALAHAN_CLEAN"] = df["PERMASALAHAN"].apply(bersih)

    tfidf = TfidfVectorizer(stop_words=stopwords_ind, max_features=2000)
    X_text = tfidf.fit_transform(df["PERMASALAHAN_CLEAN"])

    # -------------------------
    # 3. KMeans untuk keluhan
    # -------------------------
    kmeans_keluhan = KMeans(n_clusters=n_text_clusters, random_state=42, n_init=10)
    df["CLUSTER_KELUHAN"] = kmeans_keluhan.fit_predict(X_text)

    # silhouette
    sil_score = silhouette_score(X_text, df["CLUSTER_KELUHAN"])

    # kata penting per cluster
    terms = tfidf.get_feature_names_out()
    top_terms_per_cluster = {}
    for i in range(n_text_clusters):
        center = kmeans_keluhan.cluster_centers_[i]
        top = center.argsort()[-15:][::-1]
        top_terms_per_cluster[i] = [terms[t] for t in top]

    # -------------------------
    # 4. Aggregasi per kecamatan
    # -------------------------
    kec_total = df.groupby("KECAMATAN").size().reset_index(name="TOTAL_KELUHAN")
    kec_comp = df.pivot_table(
        index="KECAMATAN",
        columns="CLUSTER_KELUHAN",
        aggfunc="size",
        fill_value=0
    )
    kec_comp.columns = [f"CL_{c}" for c in kec_comp.columns]
    kec_features = kec_total.merge(kec_comp, on="KECAMATAN")

    # scaling & clustering kecamatan
    num_cols = [c for c in kec_features.columns if c != "KECAMATAN"]
    scaler = StandardScaler()
    X_kec = scaler.fit_transform(kec_features[num_cols])

    kmeans_kec = KMeans(n_clusters=n_kec_clusters, random_state=42, n_init=10)
    kec_features["CLUSTER_KECAMATAN"] = kmeans_kec.fit_predict(X_kec)
    kec_features["KECAMATAN_UP"] = kec_features["KECAMATAN"].str.upper()

    # -------------------------
    # 5. Load GeoJSON
    # -------------------------
    geo = json.load(uploaded_geojson)

    return df, kec_features, geo, sil_score, top_terms_per_cluster

# =====================================================================
# FUNGSI VISUALISASI PETA
# =====================================================================

def build_peta_cluster(df, kec_features, geo, cluster_id):
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
        kec_up = row["KECAMATAN_UP"]
        jumlah = row["JUMLAH_CLUSTER"]

        coords = get_polygon_from_geojson(geo, kec_up)
        if coords is None:
            continue

        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            fill=True,
            fill_color=get_color(jumlah),
            fill_opacity=0.8,
            color="black",
            weight=1,
            tooltip=f"{kec_up.title()}<br>Keluhan CL{cluster_id}: {int(jumlah)}"
        ).add_to(m)

    return m

def build_heatmap_cluster(df, geo, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)
    heat_points = []

    for _, row in df_cl.iterrows():
        kec = row["KECAMATAN"]
        lat0, lon0 = get_centroid_from_geojson(geo, kec)
        if lat0 is None:
            continue

        heat_points.append([
            lat0 + np.random.uniform(-0.005, 0.005),
            lon0 + np.random.uniform(-0.005, 0.005),
        ])

    if heat_points:
        HeatMap(heat_points, radius=25, blur=30).add_to(m)

    return m

CLUSTER_COLORS = {
    0: "#E53935",  # merah
    1: "#FB8C00",  # oranye
    2: "#FDD835",  # kuning
    3: "#43A047",  # hijau
    4: "#1E88E5",  # biru
}

def build_dot_cluster(df, geo, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)

    for _, row in df_cl.iterrows():
        kec = row["KECAMATAN"]
        lat0, lon0 = get_centroid_from_geojson(geo, kec)
        if lat0 is None:
            continue

        folium.CircleMarker(
            location=[
                lat0 + np.random.uniform(-0.004, 0.004),
                lon0 + np.random.uniform(-0.004, 0.004),
            ],
            radius=3,
            color=CLUSTER_COLORS.get(cluster_id, "#000000"),
            fill=True,
            fill_color=CLUSTER_COLORS.get(cluster_id, "#000000"),
            fill_opacity=0.8,
        ).add_to(m)

    return m

# =====================================================================
# VISUALISASI WORDCLOUD & BAR CHART
# =====================================================================

def show_wordcloud(df, cluster_id):
    text = " ".join(df[df["CLUSTER_KELUHAN"] == cluster_id]["PERMASALAHAN_CLEAN"])
    if not text.strip():
        st.warning(f"Tidak ada teks untuk cluster {cluster_id}")
        return

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Wordcloud Cluster {cluster_id}")
    st.pyplot(fig)

def show_bar_cluster(df, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    kec_count = df_cl["KECAMATAN"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    kec_count.plot(kind="bar", ax=ax)
    ax.set_title(f"Jumlah Keluhan per Kecamatan – Cluster {cluster_id}")
    ax.set_xlabel("Kecamatan")
    ax.set_ylabel("Jumlah Keluhan")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

def show_silhouette_plot(sil_score):
    fig, ax = plt.subplots(figsize=(4, 1.5))
    ax.barh(["Model"], [sil_score])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Nilai Silhouette (0-1)")
    ax.set_title("Kualitas Clustering (Silhouette)")
    for i, v in enumerate([sil_score]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    st.pyplot(fig)

# =====================================================================
# STREAMLIT APP
# =====================================================================

def main():
    st.title("Pemetaan Keluhan Kecamatan Bekasi")
    st.write("Visualisasi **Heatmap, Peta Poligon, Dot Density, Wordcloud**, dan kualitas clustering (TF-IDF + Silhouette).")

    with st.sidebar:
        st.header("Upload Data")
        uploaded_excel = st.file_uploader("Upload file Excel keluhan (misal: 2024.xlsx)", type=["xlsx"])
        uploaded_geojson = st.file_uploader("Upload file GeoJSON kecamatan (bekasi_kecamatan.geojson)", type=["geojson", "json"])

        st.markdown("---")
        n_text_clusters = st.slider("Jumlah cluster keluhan (TF-IDF + KMeans)", 2, 8, 5)
        n_kec_clusters = st.slider("Jumlah cluster kecamatan", 2, 8, 4)

    if not uploaded_excel or not uploaded_geojson:
        st.info("Silakan upload file Excel dan GeoJSON terlebih dahulu.")
        return

    # Proses data
    with st.spinner("Memproses data, melakukan TF-IDF & clustering..."):
        try:
            df, kec_features, geo, sil_score, top_terms_per_cluster = process_data(
                uploaded_excel,
                uploaded_geojson,
                n_text_clusters=n_text_clusters,
                n_kec_clusters=n_kec_clusters
            )
        except Exception as e:
            st.error(f"Terjadi error saat memproses data: {e}")
            return

    # ====================== SUMMARY =========================
    st.subheader("Ringkasan Data & Clustering")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Keluhan", len(df))
    with col2:
        st.metric("Jumlah Kecamatan Terdata", df["KECAMATAN"].nunique())
    with col3:
        st.metric("Silhouette Score", f"{sil_score:.3f}")

    show_silhouette_plot(sil_score)

    # ====================== TF-IDF TOP TERMS =================
    st.subheader("Kata-Kata TF-IDF Teratas per Cluster Keluhan")
    selected_cluster_tf = st.selectbox(
        "Pilih cluster untuk melihat kata penting:",
        options=sorted(top_terms_per_cluster.keys())
    )

    st.write(f"Top 15 kata pada cluster {selected_cluster_tf}:")
    st.write(", ".join(top_terms_per_cluster[selected_cluster_tf]))

    # ====================== PILIH CLUSTER UNTUK PETA =================
    st.subheader("Visualisasi Peta per Cluster Keluhan")
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        cluster_id = st.number_input(
            "Pilih ID Cluster Keluhan", min_value=0,
            max_value=n_text_clusters - 1,
            value=0,
            step=1
        )
    with c_col2:
        map_type = st.radio(
            "Jenis peta:",
            ["Peta Poligon", "Heatmap", "Dot Density"],
            horizontal=True
        )

    # ====================== TAMPILKAN PETA ====================
    st.markdown(f"### Cluster Keluhan {cluster_id} – {map_type}")

    if map_type == "Peta Poligon":
        m = build_peta_cluster(df, kec_features, geo, cluster_id)
    elif map_type == "Heatmap":
        m = build_heatmap_cluster(df, geo, cluster_id)
    else:
        m = build_dot_cluster(df, geo, cluster_id)

    st_folium(m, width=700, height=500)

    # ====================== WORDCLOUD & BAR ==================
    st.subheader(f"Wordcloud & Distribusi Keluhan – Cluster {cluster_id}")
    wc_col, bar_col = st.columns(2)
    with wc_col:
        show_wordcloud(df, cluster_id)
    with bar_col:
        show_bar_cluster(df, cluster_id)

    # ====================== DATAFRAME OPSIONAL ===============
    with st.expander("Lihat data mentah (setelah cleaning):"):
        st.dataframe(df.head(1000))

    with st.expander("Lihat fitur agregat per kecamatan:"):
        st.dataframe(kec_features)

if __name__ == "__main__":
    main()
