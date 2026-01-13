import io
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Robust import: tampilkan error jelas kalau sklearn tidak ada
# ============================================================
try:
    from sklearn.cluster import KMeans
    from sklearn_extra.cluster import KMedoids
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
except Exception as e:
    st.set_page_config(page_title="Klusterisasi Pengaduan Bekasi", page_icon="📌", layout="wide")
    st.error(
        "Gagal import library ML (scikit-learn / scikit-learn-extra).\n\n"
        "Penyebab paling sering di Streamlit Cloud: `requirements.txt` tidak terbaca (tidak di root repo) "
        "atau build dependencies gagal.\n\n"
        f"Detail error:\n`{repr(e)}`"
    )
    st.stop()

import nltk
from nltk.corpus import stopwords

import folium
from streamlit_folium import st_folium

import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Klusterisasi Pengaduan 1500-444 Bekasi",
    page_icon="📌",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📌 Klusterisasi Pengaduan Masyarakat — Bekasi (1500–444)")
st.caption("Upload data terbaru atau tambah pengaduan via form. Model final: TF-IDF + K-Means.")


# ----------------------------
# NLTK (cache + safe)
# ----------------------------
@st.cache_resource
def load_stopwords_id():
    """
    Load stopwords bahasa Indonesia secara aman.
    Di environment fresh (Streamlit Cloud), stopwords bisa belum ada.
    """
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        return stopwords.words("indonesian")
    except Exception:
        # fallback minimal kalau corpus gagal (jarang)
        return ["yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan"]


STOPWORDS_ID = load_stopwords_id()


# ----------------------------
# Helpers: Cleaning & Mapping
# ----------------------------
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
    "Pejuang": "Medan Satria"
}

REQUIRED_COLS = ["ALAMAT", "PERMASALAHAN"]


def is_noise_address(a: str) -> bool:
    a = str(a).lower().strip()
    if re.fullmatch(r"[0-9\-\+\(\) ]{7,}", a):
        return True
    bad = ["pt", "cv", "tbk", "bank", "bca", "kantor", "industri", "factory"]
    return any(b in a for b in bad)


def detect_kec_from_text(a: str):
    a_low = str(a).lower()
    for k in KEC_LIST:
        if k.lower() in a_low:
            return k
    return None


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan. Wajib ada: {REQUIRED_COLS}")

    # drop baris kosong
    df = df.dropna(subset=REQUIRED_COLS).copy()

    # Cleaning alamat
    df = df[~df["ALAMAT"].astype(str).apply(is_noise_address)].copy()
    df["ALAMAT"] = df["ALAMAT"].astype(str).str.title().str.strip()

    # Normalisasi kecamatan
    df["KECAMATAN"] = df["ALAMAT"].map(MAPPING)
    df["KECAMATAN"] = df["KECAMATAN"].fillna(df["ALAMAT"].apply(detect_kec_from_text))
    df = df.dropna(subset=["KECAMATAN"]).copy()

    # Cleaning teks keluhan
    df["PERMASALAHAN_CLEAN"] = df["PERMASALAHAN"].apply(clean_text)

    # drop teks kosong setelah cleaning
    df = df[df["PERMASALAHAN_CLEAN"].astype(str).str.len() > 0].copy()

    # Upper for GeoJSON match
    df["KECAMATAN_UP"] = df["KECAMATAN"].astype(str).str.strip().str.upper()

    if len(df) == 0:
        raise ValueError("Data habis setelah preprocessing (alamat/noise/kecamatan/teks kosong). Cek isi XLSX kamu.")

    return df


# ----------------------------
# Modeling
# ----------------------------
@st.cache_data(show_spinner=False)
def run_model(df: pd.DataFrame, k_keluhan: int = 5):
    # TF-IDF
    tfidf = TfidfVectorizer(stop_words=STOPWORDS_ID, max_features=2000)
    X_text = tfidf.fit_transform(df["PERMASALAHAN_CLEAN"])

    # K-Means (final) - FIX: n_init pakai int (lebih aman)
    kmeans_keluhan = KMeans(n_clusters=k_keluhan, random_state=42, n_init=10)
    labels_kmeans = kmeans_keluhan.fit_predict(X_text)

    sil = None
    if len(np.unique(labels_kmeans)) > 1:
        sil = float(silhouette_score(X_text, labels_kmeans))

    # K-Medoids (opsional pembanding)
    kmedoids_keluhan = KMedoids(
        n_clusters=k_keluhan,
        metric="cosine",
        init="k-medoids++",
        random_state=42
    )
    labels_kmedoids = kmedoids_keluhan.fit_predict(X_text)

    out = df.copy()
    out["CLUSTER_KELUHAN"] = labels_kmeans
    out["CLUSTER_KELUHAN_MEDOID"] = labels_kmedoids

    # Top terms per cluster (KMeans)
    terms = tfidf.get_feature_names_out()
    top_terms = {}
    centers = kmeans_keluhan.cluster_centers_
    for i in range(k_keluhan):
        center = centers[i]
        idx = center.argsort()[-15:][::-1]
        top_terms[i] = [terms[j] for j in idx]

    return out, X_text, tfidf, sil, top_terms


@st.cache_data(show_spinner=False)
def build_kecamatan_features(df_modeled: pd.DataFrame, k_keluhan: int = 5):
    kec_total = df_modeled.groupby("KECAMATAN").size().reset_index(name="TOTAL_KELUHAN")

    kec_comp = df_modeled.pivot_table(
        index="KECAMATAN",
        columns="CLUSTER_KELUHAN",
        aggfunc="size",
        fill_value=0
    )

    # pastikan kolom CL_0..CL_{k-1} lengkap
    for c in range(k_keluhan):
        if c not in kec_comp.columns:
            kec_comp[c] = 0
    kec_comp = kec_comp[[c for c in range(k_keluhan)]]
    kec_comp.columns = [f"CL_{c}" for c in range(k_keluhan)]

    kec_features = kec_total.merge(kec_comp, on="KECAMATAN")

    # clustering kecamatan (k=4)
    num_cols = [c for c in kec_features.columns if c != "KECAMATAN"]
    scaler = StandardScaler()
    X_kec = scaler.fit_transform(kec_features[num_cols])

    # FIX: n_init pakai int (lebih aman)
    kmeans_kec = KMeans(n_clusters=4, random_state=42, n_init=10)
    kec_features["CLUSTER_KECAMATAN"] = kmeans_kec.fit_predict(X_kec)

    kmedoids_kec = KMedoids(n_clusters=4, metric="euclidean", random_state=42)
    kec_features["CLUSTER_KECAMATAN_KMEDOID"] = kmedoids_kec.fit_predict(X_kec)

    kec_features["KECAMATAN_UP"] = kec_features["KECAMATAN"].astype(str).str.strip().str.upper()
    return kec_features


# ----------------------------
# GeoJSON helpers
# ----------------------------
def extract_first_ring(geom):
    t = geom.get("type")
    coords = geom.get("coordinates")
    if t == "Polygon":
        return coords[0]
    if t == "MultiPolygon":
        return coords[0][0]
    return None


def build_geo_map(geojson_obj):
    geo_map = {}
    for feat in geojson_obj.get("features", []):
        props = feat.get("properties", {})
        name = str(props.get("KECAMATAN", "")).strip().upper()
        ring = extract_first_ring(feat.get("geometry", {}))
        if name and ring:
            geo_map[name] = ring
    return geo_map


def draw_polygon(m, coords, color, tooltip):
    folium.Polygon(
        locations=[(lat, lon) for lon, lat in coords],
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        color="black",
        weight=1,
        tooltip=tooltip
    ).add_to(m)


def map_cluster_kecamatan(kec_features, geo_map, cluster_col, title):
    colors = ["#E53935", "#FB8C00", "#FDD835", "#43A047", "#1E88E5", "#8E24AA"]
    m = folium.Map(location=[-6.25, 107.03], zoom_start=12, tiles="cartodbpositron")

    for _, row in kec_features.iterrows():
        kec = row["KECAMATAN_UP"]
        cl = int(row[cluster_col])
        if kec not in geo_map:
            continue
        draw_polygon(
            m,
            geo_map[kec],
            colors[cl % len(colors)],
            f"{kec}<br>{title}: {cl}"
        )
    return m


def map_keluhan_cluster(df_modeled, kec_features, geo_map, cluster_id):
    df_cl = df_modeled[df_modeled["CLUSTER_KELUHAN"] == cluster_id]
    cl_kec = df_cl.groupby("KECAMATAN").size().reset_index(name="JUMLAH")
    kec_cl = kec_features.merge(cl_kec, on="KECAMATAN", how="left").fillna(0)

    m = folium.Map(location=[-6.25, 107.03], zoom_start=12, tiles="cartodbpositron")

    def get_color(v):
        v = int(v)
        if v == 0:
            return "#E0E0E0"
        if v <= 5:
            return "#FFF176"
        if v <= 15:
            return "#FFB74D"
        return "#E53935"

    for _, row in kec_cl.iterrows():
        kec = row["KECAMATAN_UP"]
        jumlah = int(row["JUMLAH"])
        if kec not in geo_map:
            continue
        draw_polygon(
            m,
            geo_map[kec],
            get_color(jumlah),
            f"{kec}<br>Keluhan CL{cluster_id}: {jumlah}"
        )
    return m


# ----------------------------
# Sidebar: Data input & update
# ----------------------------
st.sidebar.header("⚙️ Data & Pengaturan")

uploaded_xlsx = st.sidebar.file_uploader("Upload data XLSX (2024.xlsx)", type=["xlsx"])
uploaded_geojson = st.sidebar.file_uploader("Upload GeoJSON kecamatan (opsional)", type=["geojson", "json"])

k_keluhan = st.sidebar.number_input("Jumlah cluster keluhan (K)", min_value=2, max_value=10, value=5, step=1)
run_btn = st.sidebar.button("🔁 Proses / Update Model", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("➕ Tambah Pengaduan (tanpa edit XLSX)")
with st.sidebar.form("add_row_form", clear_on_submit=True):
    alamat_new = st.text_input("Alamat", placeholder="Contoh: Jl. ..., Bekasi Timur")
    permasalahan_new = st.text_area("Permasalahan", placeholder="Contoh: lampu jalan mati...")
    submit_add = st.form_submit_button("Tambah", use_container_width=True)


def load_from_upload(file) -> pd.DataFrame:
    return pd.read_excel(file)


if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if uploaded_xlsx is not None:
    st.session_state.raw_df = load_from_upload(uploaded_xlsx)

# tambah row via form
if submit_add:
    if st.session_state.raw_df is None:
        st.warning("Upload XLSX dulu, baru bisa tambah pengaduan via form.")
    else:
        new_row = {"ALAMAT": alamat_new, "PERMASALAHAN": permasalahan_new}
        st.session_state.raw_df = pd.concat(
            [st.session_state.raw_df, pd.DataFrame([new_row])],
            ignore_index=True
        )
        st.success("Pengaduan berhasil ditambahkan ke dataset (sementara). Klik 'Proses / Update Model'.")


# ----------------------------
# Main content
# ----------------------------
if st.session_state.raw_df is None:
    st.info("Upload file XLSX terlebih dahulu di sidebar. Kolom wajib: ALAMAT, PERMASALAHAN.")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in st.session_state.raw_df.columns]
if missing:
    st.error(f"Kolom wajib tidak lengkap: {missing}. Pastikan XLSX berisi kolom {REQUIRED_COLS}.")
    st.stop()

# run pipeline
if run_btn or ("modeled_df" not in st.session_state):
    with st.spinner("Memproses data (cleaning → TF-IDF → K-Means)…"):
        try:
            pre = preprocess_df(st.session_state.raw_df)
            modeled, X_text, tfidf_obj, sil, top_terms = run_model(pre, k_keluhan=int(k_keluhan))
            kec_features = build_kecamatan_features(modeled, k_keluhan=int(k_keluhan))

            st.session_state.modeled_df = modeled
            st.session_state.kec_features = kec_features
            st.session_state.sil = sil
            st.session_state.top_terms = top_terms
        except Exception as e:
            st.error(f"Gagal memproses: {e}")
            st.stop()

modeled_df = st.session_state.modeled_df
kec_features = st.session_state.kec_features
sil = st.session_state.sil
top_terms = st.session_state.top_terms

# Export
st.sidebar.divider()
st.sidebar.subheader("⬇️ Export")
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    st.session_state.raw_df.to_excel(writer, index=False, sheet_name="DATA_RAW")
    modeled_df.to_excel(writer, index=False, sheet_name="DATA_CLUSTER")
    kec_features.to_excel(writer, index=False, sheet_name="KECAMATAN_CLUSTER")
buffer.seek(0)

st.sidebar.download_button(
    "Download XLSX (Raw + Cluster + Kecamatan)",
    data=buffer,
    file_name=f"hasil_klusterisasi_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Data (raw)", len(st.session_state.raw_df))
c2.metric("Total Data (clean)", len(modeled_df))
c3.metric("Jumlah Kecamatan", modeled_df["KECAMATAN"].nunique())
c4.metric("Silhouette (K-Means)", "-" if sil is None else f"{sil:.4f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Data", "🧠 Modeling", "🗺️ Peta", "📊 Insight"])

with tab1:
    st.subheader("Preview Data")
    st.write("Data setelah preprocessing (cleaning alamat, deteksi kecamatan, cleaning teks).")
    st.dataframe(modeled_df.head(50), use_container_width=True)

    st.subheader("Ringkasan Keluhan per Kecamatan")
    ringkas = modeled_df["KECAMATAN"].value_counts().reset_index()
    ringkas.columns = ["KECAMATAN", "TOTAL"]
    st.dataframe(ringkas, use_container_width=True)

with tab2:
    st.subheader("Top Terms per Cluster (K-Means)")
    cols = st.columns(min(int(k_keluhan), 5))
    for i in range(int(k_keluhan)):
        with cols[i % len(cols)]:
            st.markdown(f"**Cluster {i}**")
            st.write(", ".join(top_terms.get(i, [])[:12]))

    st.subheader("Distribusi Cluster Keluhan")
    vc = modeled_df["CLUSTER_KELUHAN"].value_counts().sort_index()
    fig = plt.figure()
    vc.plot(kind="bar", grid=True)
    plt.title("Jumlah Data per Cluster Keluhan (K-Means)")
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Peta (Folium) — Upload GeoJSON untuk menampilkan polygon")
    if uploaded_geojson is None:
        st.info("Upload GeoJSON kecamatan di sidebar (opsional). Setelah itu peta polygon akan tampil.")
    else:
        try:
            geo = json.load(uploaded_geojson)
            geo_map = build_geo_map(geo)

            if not geo_map:
                st.warning("GeoJSON terbaca, tapi mapping 'properties.KECAMATAN' tidak ditemukan / kosong.")
            else:
                left, right = st.columns(2)

                with left:
                    st.markdown("**Cluster Kecamatan (K-Means)**")
                    m1 = map_cluster_kecamatan(
                        kec_features, geo_map, "CLUSTER_KECAMATAN", "Cluster Kecamatan (K-Means)"
                    )
                    st_folium(m1, width=None, height=520)

                with right:
                    st.markdown("**Cluster Kecamatan (K-Medoids)**")
                    m2 = map_cluster_kecamatan(
                        kec_features, geo_map, "CLUSTER_KECAMATAN_KMEDOID", "Cluster Kecamatan (K-Medoids)"
                    )
                    st_folium(m2, width=None, height=520)

                st.divider()
                st.markdown("**Peta Sebaran Keluhan per Cluster (K-Means)**")
                cl = st.selectbox("Pilih Cluster Keluhan", list(range(int(k_keluhan))), index=0)
                m3 = map_keluhan_cluster(modeled_df, kec_features, geo_map, int(cl))
                st_folium(m3, width=None, height=520)

        except Exception as e:
            st.error(f"Gagal membaca/menampilkan GeoJSON: {e}")

with tab4:
    st.subheader("Wordcloud & Bar per Cluster")
    cl = st.selectbox("Pilih Cluster", list(range(int(k_keluhan))), index=0, key="insight_cluster")

    text = " ".join(
        modeled_df[modeled_df["CLUSTER_KELUHAN"] == int(cl)]["PERMASALAHAN_CLEAN"]
        .astype(str).tolist()
    ).strip()

    if not text:
        st.info("Tidak ada teks di cluster ini.")
    else:
        wc = WordCloud(width=900, height=380, background_color="white").generate(text)
        fig_wc = plt.figure(figsize=(10, 4))
        plt.imshow(wc)
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig_wc)

    st.subheader("Keluhan per Kecamatan (Cluster terpilih)")
    fig2 = plt.figure()
    (modeled_df[modeled_df["CLUSTER_KELUHAN"] == int(cl)]["KECAMATAN"]
        .value_counts()
        .plot(kind="bar", grid=True))
    plt.title(f"Keluhan per Kecamatan — Cluster {cl}")
    plt.xlabel("Kecamatan")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    st.pyplot(fig2)
