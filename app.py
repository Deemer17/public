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

# -----------------------------------------------------------
# NLTK SETUP
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# BASIC CLEANING FUNCTIONS
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# KECAMATAN CONFIG
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# GEOJSON UTIL
# -----------------------------------------------------------
def get_polygon_from_geojson(geo, kecamatan_up):
    for feat in geo["features"]:
        if feat["properties"]["KECAMATAN"].strip().upper() == kecamatan_up.strip().upper():
            return feat["geometry"]["coordinates"][0]
    return None

def get_centroid_from_geojson(geo, kec):
    for feat in geo["features"]:
        if feat["properties"]["KECAMATAN"].title() == kec.title():
            coords = np.array(feat["geometry"]["coordinates"][0])
            lon0 = coords[:, 0].mean()
            lat0 = coords[:, 1].mean()
            return lat0, lon0
    return None, None


# -----------------------------------------------------------
# PROCESSING PIPELINE (CACHED)
# -----------------------------------------------------------
@st.cache_data(show_spinner=True)
def process_data(uploaded_excel, uploaded_geojson, n_text_clusters=5, n_kec_clusters=4):

    df = pd.read_excel(uploaded_excel)

    if "ALAMAT" not in df.columns or "PERMASALAHAN" not in df.columns:
        raise ValueError("File Excel wajib memiliki kolom 'ALAMAT' dan 'PERMASALAHAN'.")

    # cleaning alamat
    df = df[~df["ALAMAT"].astype(str).apply(is_noise)].copy()
    df["ALAMAT"] = df["ALAMAT"].astype(str).str.title().str.strip()
    df["KECAMATAN"] = df["ALAMAT"].apply(detect_kec)
    df = df.dropna(subset=["KECAMATAN"])
    df["KECAMATAN_UP"] = df["KECAMATAN"].str.upper()

    # text cleaning + TF-IDF
    df["PERMASALAHAN_CLEAN"] = df["PERMASALAHAN"].apply(bersih)
    tfidf = TfidfVectorizer(stop_words=stopwords_ind, max_features=2000)
    X_text = tfidf.fit_transform(df["PERMASALAHAN_CLEAN"])

    # clustering keluhan
    km_text = KMeans(n_clusters=n_text_clusters, random_state=42, n_init=10)
    df["CLUSTER_KELUHAN"] = km_text.fit_predict(X_text)

    # silhouette score
    sil_score = silhouette_score(X_text, df["CLUSTER_KELUHAN"])

    # top terms per cluster
    terms = tfidf.get_feature_names_out()
    term_top = {}
    for i in range(n_text_clusters):
        center = km_text.cluster_centers_[i]
        top = center.argsort()[-15:][::-1]
        term_top[i] = [terms[t] for t in top]

    # agregasi per kecamatan
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

    # clustering kecamatan
    scaler = StandardScaler()
    X_kec = scaler.fit_transform(kec_features.drop(columns=["KECAMATAN", "KECAMATAN_UP"]))
    km_kec = KMeans(n_clusters=n_kec_clusters, random_state=42, n_init=10)
    kec_features["CLUSTER_KECAMATAN"] = km_kec.fit_predict(X_kec)

    # geojson
    geo = json.load(uploaded_geojson)

    return df, kec_features, geo, sil_score, term_top


# -----------------------------------------------------------
# JITTER (STABIL, ANTI FLICKER)
# -----------------------------------------------------------
def jitter_point(lat, lon, index):
    np.random.seed(index)
    lat_j = lat + np.random.uniform(-0.003, 0.003)
    lon_j = lon + np.random.uniform(-0.003, 0.003)
    return lat_j, lon_j


# -----------------------------------------------------------
# CACHED POINTS (DOT & HEATMAP)
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# MAP BUILDERS
# -----------------------------------------------------------
DOT_COLORS = ["#E53935", "#FB8C00", "#FDD835", "#43A047", "#1E88E5", "#8E24AA", "#00897B", "#3949AB"]

def map_dot(df, geo, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    points = get_dot_points(df_cl, geo)

    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)
    color = DOT_COLORS[cluster_id % len(DOT_COLORS)]

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

    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)
    if points:
        HeatMap(points, radius=25, blur=30).add_to(m)
    return m


def map_polygon(df, kec_features, geo, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    group = df_cl.groupby("KECAMATAN").size().reset_index(name="JML")

    kec_cl = kec_features.merge(group, on="KECAMATAN", how="left")
    kec_cl["JML"] = kec_cl["JML"].fillna(0)

    m = folium.Map(location=[-6.25, 107.03], zoom_start=12)

    def color(v):
        if v == 0:
            return "#E0E0E0"
        if v <= 5:
            return "#FFF176"
        if v <= 15:
            return "#FFB74D"
        return "#E53935"

    for _, row in kec_cl.iterrows():
        coords = get_polygon_from_geojson(geo, row["KECAMATAN_UP"])
        if coords is None:
            continue
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            fill=True,
            fill_color=color(row["JML"]),
            color="#555555",
            weight=1,
            fill_opacity=0.8,
            tooltip=f"{row['KECAMATAN']} – {int(row['JML'])} keluhan"
        ).add_to(m)
    return m


# -----------------------------------------------------------
# VISUAL – WORDCLOUD & BAR
# -----------------------------------------------------------
def show_wordcloud(df, cluster_id):
    text = " ".join(df[df["CLUSTER_KELUHAN"] == cluster_id]["PERMASALAHAN_CLEAN"])
    if not text.strip():
        st.warning("Tidak ada teks pada cluster ini.")
        return

    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def show_bar(df, cluster_id):
    df_cl = df[df["CLUSTER_KELUHAN"] == cluster_id]
    count = df_cl["KECAMATAN"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    count.plot(kind="bar", ax=ax)
    ax.set_title(f"Distribusi Keluhan per Kecamatan – Cluster {cluster_id}")
    ax.set_xlabel("Kecamatan")
    ax.set_ylabel("Jumlah Keluhan")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------------------------------------
# CUSTOM CSS (CLEAN UI)
# -----------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* ruang konten lebih lebar */
        .main .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* judul utama */
        h1 {
            font-weight: 600 !important;
        }

        /* card metric */
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 0.75rem;
            border: 1px solid #E5E5E5;
            background: #FFFFFF;
            box-shadow: 0px 1px 3px rgba(15, 23, 42, 0.08);
        }

        .metric-label {
            font-size: 0.8rem;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .metric-value {
            font-size: 1.4rem;
            font-weight: 600;
            color: #111827;
        }

        .metric-sub {
            font-size: 0.8rem;
            color: #9CA3AF;
        }

        /* tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #F9FAFB;
            border-radius: 0.75rem;
            padding-top: 0.4rem;
            padding-bottom: 0.4rem;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }

        .stTabs [aria-selected="true"] {
            background-color: #111827 !important;
            color: #F9FAFB !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------
# STREAMLIT MAIN APP
# -----------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Dashboard Keluhan Kota Bekasi",
        page_icon="📍",
        layout="wide",
    )
    inject_css()

    st.title("📍 Dashboard Keluhan Kota Bekasi")
    st.caption("Analisis clustering keluhan warga dan visualisasi sebaran spasial per kecamatan.")

    with st.sidebar:
        st.header("⚙️ Pengaturan Data")
        excel = st.file_uploader("Upload Excel (misal: 2024.xlsx)", type=["xlsx"])
        geo = st.file_uploader("Upload GeoJSON (bekasi_kecamatan.geojson)", type=["geojson", "json"])

        st.markdown("---")
        st.header("🔢 Parameter Clustering")
        n_text_clusters = st.slider("Jumlah cluster keluhan", 2, 8, 5)
        n_kec_clusters = st.slider("Jumlah cluster kecamatan", 2, 8, 4)

        st.markdown("---")
        st.caption("Pastikan kolom: **ALAMAT** & **PERMASALAHAN** ada di file Excel.")

    if not excel or not geo:
        st.info("Silakan upload file Excel dan GeoJSON melalui sidebar untuk memulai.")
        return

    with st.spinner("Memproses data, melakukan TF-IDF dan clustering..."):
        try:
            df, kec_features, geojson, sil, terms = process_data(
                excel, geo, n_text_clusters, n_kec_clusters
            )
        except Exception as e:
            st.error(f"Terjadi error saat memproses data: {e}")
            return

    # ================= OVERVIEW =================
    tab_overview, tab_map, tab_cluster, tab_data = st.tabs(
        ["📊 Overview", "🗺️ Peta", "🔍 Detail Cluster", "📑 Data"]
    )

    # ---- OVERVIEW TAB ----
    with tab_overview:
        st.subheader("Ringkasan Analisis")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">Total Keluhan</div>
                  <div class="metric-value">{len(df):,}</div>
                  <div class="metric-sub">Baris data setelah cleaning</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">Jumlah Kecamatan</div>
                  <div class="metric-value">{df["KECAMATAN"].nunique()}</div>
                  <div class="metric-sub">Terdeteksi dari kolom alamat</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">Silhouette Score</div>
                  <div class="metric-value">{sil:.3f}</div>
                  <div class="metric-sub">Kualitas cluster keluhan (0–1)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Kualitas Clustering (Silhouette)")
        fig, ax = plt.subplots(figsize=(4, 1.6))
        ax.barh(["Model"], [sil], color="#111827")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Nilai Silhouette")
        for i, v in enumerate([sil]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center")
        st.pyplot(fig)

        st.markdown("### Kata-Kata TF-IDF Teratas per Cluster")
        pilih = st.selectbox("Pilih cluster keluhan:", sorted(terms.keys()))
        st.write(", ".join(terms[pilih]))

    # ---- MAP TAB ----
    with tab_map:
        st.subheader("Visualisasi Peta per Cluster Keluhan")

        col_left, col_right = st.columns([1, 2])
        with col_left:
            cluster_id = st.number_input(
                "Cluster ID keluhan",
                min_value=0,
                max_value=n_text_clusters - 1,
                value=0,
                step=1,
            )
            map_type = st.radio(
                "Jenis peta",
                ["Peta Poligon", "Heatmap", "Dot Density"],
                index=0,
            )

            st.caption(
                "• **Poligon** → intensitas keluhan per kecamatan\n"
                "• **Heatmap** → kepadatan sebaran keluhan\n"
                "• **Dot** → titik-titik keluhan (dengan jitter stabil)"
            )

        with col_right:
            if map_type == "Peta Poligon":
                map_obj = map_polygon(df, kec_features, geojson, cluster_id)
            elif map_type == "Heatmap":
                map_obj = map_heat(df, geojson, cluster_id)
            else:
                map_obj = map_dot(df, geojson, cluster_id)

            st_folium(map_obj, width=750, height=520)

    # ---- CLUSTER DETAIL TAB ----
    with tab_cluster:
        st.subheader("Detail Cluster Keluhan")

        cl_detail = st.number_input(
            "Pilih cluster untuk dianalisis",
            min_value=0,
            max_value=n_text_clusters - 1,
            value=0,
            step=1,
        )

        col_wc, col_bar = st.columns(2)
        with col_wc:
            st.markdown("##### Wordcloud Keluhan")
            show_wordcloud(df, cl_detail)

        with col_bar:
            st.markdown("##### Distribusi Keluhan per Kecamatan")
            show_bar(df, cl_detail)

    # ---- DATA TAB ----
    with tab_data:
        st.subheader("Data yang Digunakan")

        st.markdown("#### Sampel Data Keluhan (setelah cleaning)")
        st.dataframe(df.head(500))

        st.markdown("#### Fitur Agregat per Kecamatan")
        st.dataframe(kec_features)


if __name__ == "__main__":
    main()
