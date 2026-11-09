import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Segmentasi Blok P2 Estate", page_icon="🌴",
                   layout="wide", initial_sidebar_state="expanded")

DEFAULT_FILE = "Dataset Primer.xlsx"
HIJAU, ORANYE, MERAH = "#1B9E5A", "#F39C12", "#E74C3C"

# ───────────────────────── Styling ─────────────────────────
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#0F5E36 0%,#1B9E5A 60%,#7BC47F 100%);
       color:#fff; padding: 26px 32px; border-radius: 16px; margin-bottom: 18px;}
.hero h1 {color:#fff; font-size: 1.9rem; margin:0 0 6px 0; padding:0;}
.hero p  {color:#E8F5EC; margin:0; font-size: .98rem;}
.hero .tag {display:inline-block; background:rgba(255,255,255,.18); border-radius:20px;
            padding:3px 12px; margin:10px 8px 0 0; font-size:.8rem;}
[data-testid="stMetric"] {background: rgba(27,158,90,.08); border:1px solid rgba(27,158,90,.25);
            border-radius: 12px; padding: 14px 16px;}
[data-testid="stMetricLabel"] p {font-size:.85rem; opacity:.8;}
.seg {border-radius: 14px; padding: 16px 18px; border-left: 7px solid var(--c);
      background: rgba(128,128,128,.07); height: 100%;}
.seg h4 {margin: 0 0 2px 0; font-size: 1.05rem;}
.seg .n {font-size: 1.9rem; font-weight: 700; line-height: 1.1; color: var(--c);}
.seg .sub {font-size:.82rem; opacity:.7; margin-bottom:8px;}
.seg table {width:100%; font-size:.86rem; border-collapse:collapse;}
.seg td {padding:2px 0;} .seg td:last-child {text-align:right; font-weight:600;}
.seg .rek {margin-top:10px; font-size:.84rem; padding-top:8px; border-top:1px dashed rgba(128,128,128,.4);}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 16px;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ───────────────────────── Helper (kompatibel lintas versi Streamlit) ─────────────────────────
def tampil_grafik(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def tampil_tabel(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def peta_warna(urutan_label):
    sedang_palet = ["#F39C12", "#F1C40F", "#E67E22", "#D4AC0D", "#CA6F1E"]
    warna, i = {}, 0
    for lb in urutan_label:
        if lb.endswith("Tinggi"):
            warna[lb] = HIJAU
        elif lb.endswith("Rendah"):
            warna[lb] = MERAH
        else:
            warna[lb] = sedang_palet[i % len(sedang_palet)]
            i += 1
    return warna


REKOMENDASI = {
    "Tinggi": "Produktif & mendekati/melampaui target → pertahankan pemeliharaan.",
    "Sedang": "Kinerja menengah → monitor rutin dan cari peluang perbaikan.",
    "Rendah": "Di bawah target → prioritas evaluasi agronomis & manajemen.",
}


def kartu_segmen(label, row, warna):
    jenis = label.split(": ")[-1].replace("Kinerja ", "")
    st.markdown(f"""
<div class="seg" style="--c:{warna}">
  <h4>{label}</h4>
  <div class="n">{int(row['Jumlah_Blok'])} blok</div>
  <div class="sub">{row['Persen']:.1f}% dari total blok</div>
  <table>
    <tr><td>Yield Aktual</td><td>{row['Yield_Aktual']:.2f} ton/ha</td></tr>
    <tr><td>CV Produksi</td><td>{row['CV_Produksi']:.2f} %</td></tr>
    <tr><td>Achievement Rate</td><td>{row['Achievement_Rate']:.2f} %</td></tr>
    <tr><td>Yield Gap</td><td>{row['Yield_Gap']:.2f} ton/ha</td></tr>
  </table>
  <div class="rek">{REKOMENDASI.get(jenis, '')}</div>
</div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Memproses data & melatih K-Means...")
def proses(file_bytes, nama, k_manual, k_min, k_max):
    if nama.lower().endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    df_raw, hilang = pl.cek_kolom(df_raw)
    if hilang:
        return None, hilang, df_raw
    df = pl.bersihkan_data(df_raw)
    res = pl.jalankan_segmentasi(pl.buat_fitur_blok(df), k_min=k_min, k_max=k_max,
                                 k_manual=k_manual)
    return res, [], df_raw


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("### 🌴 Segmentasi Blok Sawit")
    st.caption("P2 Estate, Sumatera Utara · K-Means Clustering")
    upload = st.file_uploader("📂 Upload dataset (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
    if upload is not None:
        file_bytes, nama = upload.getvalue(), upload.name
    elif os.path.exists(DEFAULT_FILE):
        with open(DEFAULT_FILE, "rb") as f:
            file_bytes, nama = f.read(), DEFAULT_FILE
        st.success(f"Memakai file bawaan: {DEFAULT_FILE}")
    else:
        file_bytes, nama = None, None

    st.markdown("---")
    st.markdown("**⚙️ Pengaturan model**")
    k_min, k_max = st.slider("Rentang kandidat K", 2, 10, (2, 8))
    mode_k = st.radio("Penentuan K", ["Otomatis (Elbow vs Silhouette)", "Manual"])
    k_manual = None
    if mode_k == "Manual":
        k_manual = int(st.number_input("Nilai K", min_value=k_min, max_value=k_max,
                                       value=min(max(3, k_min), k_max), step=1))
    st.markdown("---")
    st.caption("Daniel Kristiano Dali · NPM 202210715046")

# ───────────────────────── Header ─────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌴 Segmentasi Kinerja Blok Perkebunan Kelapa Sawit</h1>
  <p>Mengelompokkan blok kebun berdasarkan pola produksi FFB dan efisiensi operasional
  dalam satu model K-Means terpadu.</p>
  <span class="tag">Yield Aktual</span><span class="tag">CV Produksi</span>
  <span class="tag">Achievement Rate</span><span class="tag">Yield Gap</span>
</div>
""", unsafe_allow_html=True)

if file_bytes is None:
    st.info("⬅️ **Mulai dengan meng-upload dataset di sidebar.**")
    st.markdown("Kolom yang dibutuhkan: " + ", ".join(f"`{c}`" for c in pl.KOLOM_WAJIB))
    st.stop()

try:
    res, hilang, df_raw = proses(file_bytes, nama, k_manual, k_min, k_max)
except Exception as e:
    st.error(f"Gagal memproses file: {e}")
    st.stop()

if hilang:
    st.error("Kolom berikut tidak ditemukan di dataset: " + ", ".join(hilang))
    st.write("Kolom yang terbaca:", list(df_raw.columns))
    st.stop()

dfc, k_opt = res["df_cluster"], res["k_opt"]
urutan = res["urutan_label"]
warna = peta_warna(urutan)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Ringkasan", "🎯 Penentuan K", "🗺️ Visualisasi", "✅ Evaluasi", "📥 Data & Prediksi"])

# ───────────────────────── Tab 1 ─────────────────────────
with tab1:
    c = st.columns(4)
    c[0].metric("Jumlah blok", len(dfc))
    c[1].metric("K terpilih", k_opt)
    c[2].metric("Silhouette Score", f"{res['silhouette']:.4f}")
    c[3].metric("Davies-Bouldin Index", f"{res['dbi']:.4f}")

    st.markdown("#### Profil tiap segmen")
    profil = dfc.groupby("Label_Segmen").agg(
        Jumlah_Blok=("Block", "count"), Yield_Aktual=("Yield_Aktual", "mean"),
        CV_Produksi=("CV_Produksi", "mean"), Achievement_Rate=("Achievement_Rate", "mean"),
        Yield_Gap=("Yield_Gap", "mean")).reindex(urutan)
    profil["Persen"] = profil["Jumlah_Blok"] / len(dfc) * 100

    kolom_kartu = st.columns(len(urutan) if len(urutan) <= 4 else 4)
    for i, lb in enumerate(urutan):
        with kolom_kartu[i % len(kolom_kartu)]:
            kartu_segmen(lb, profil.loc[lb], warna[lb])
            st.write("")

    c1, c2 = st.columns(2)
    with c1:
        dist = dfc["Label_Segmen"].value_counts().reindex(urutan).reset_index()
        dist.columns = ["Segmen", "Jumlah Blok"]
        fig = px.pie(dist, names="Segmen", values="Jumlah Blok", color="Segmen",
                     color_discrete_map=warna, hole=0.5, title="Proporsi blok per segmen")
        tampil_grafik(fig)
    with c2:
        pd_div = dfc.groupby(["Divisi", "Label_Segmen"]).size().reset_index(name="Jumlah")
        fig = px.bar(pd_div, x="Divisi", y="Jumlah", color="Label_Segmen",
                     color_discrete_map=warna, category_orders={"Label_Segmen": urutan},
                     title="Sebaran segmen per Divisi")
        tampil_grafik(fig)

    pm = dfc.groupby("Label_Segmen")[pl.FITUR].mean().reindex(urutan)
    rentang = (pm.max() - pm.min()).replace(0, 1)
    fig = px.imshow((pm - pm.min()) / rentang, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdYlGn", title="Heatmap profil cluster (dinormalisasi 0–1)")
    tampil_grafik(fig)

# ───────────────────────── Tab 2 ─────────────────────────
with tab2:
    dm = res["df_metrik"]
    st.success(f"**K terpilih = {k_opt}.** {res['alasan_k']}")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=dm.index, y=dm["WCSS"], mode="lines+markers",
                                   line=dict(color="#2E86C1", width=3)))
        fig.add_vline(x=k_opt, line_dash="dash", line_color=HIJAU, annotation_text=f"K={k_opt}")
        fig.update_layout(title="Elbow Method (WCSS)", xaxis_title="K", yaxis_title="WCSS")
        tampil_grafik(fig)
    with c2:
        fig = go.Figure(go.Scatter(x=dm.index, y=dm["Silhouette"], mode="lines+markers",
                                   line=dict(color=MERAH, width=3)))
        fig.add_vline(x=k_opt, line_dash="dash", line_color=HIJAU, annotation_text=f"K={k_opt}")
        fig.update_layout(title="Silhouette Score", xaxis_title="K", yaxis_title="Silhouette")
        tampil_grafik(fig)
    st.caption(f"Elbow menunjuk K={res['elbow_k']} · Silhouette menunjuk K={res['sil_k']}. "
               "DBI tidak dipakai menentukan K, hanya untuk evaluasi.")
    tampil_tabel(dm.round(4))

# ───────────────────────── Tab 3 ─────────────────────────
with tab3:
    var, cp = res["var_exp"], res["cent_pca"]
    fig = px.scatter(dfc, x="PCA1", y="PCA2", color="Label_Segmen", color_discrete_map=warna,
                     category_orders={"Label_Segmen": urutan},
                     hover_data=["Block", "Divisi"] + pl.FITUR,
                     title=f"PCA 2D — K={k_opt} (PC1+PC2 = {var.sum():.1f}% variansi)")
    fig.update_traces(marker=dict(size=9, line=dict(width=0.6, color="white")))
    fig.add_trace(go.Scatter(x=cp[:, 0], y=cp[:, 1], mode="markers", name="Centroid",
                             marker=dict(symbol="x", size=15, color="black")))
    fig.update_xaxes(title=f"PC1 ({var[0]:.1f}%)")
    fig.update_yaxes(title=f"PC2 ({var[1]:.1f}%)")
    tampil_grafik(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(dfc, x="Yield_Aktual", y="Achievement_Rate", color="Label_Segmen",
                         color_discrete_map=warna, category_orders={"Label_Segmen": urutan},
                         hover_data=["Block", "Divisi"], title="Yield Aktual vs Achievement Rate")
        fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Target 100%")
        tampil_grafik(fig)
    with c2:
        fig = px.scatter(dfc, x="CV_Produksi", y="Yield_Gap", color="Label_Segmen",
                         color_discrete_map=warna, category_orders={"Label_Segmen": urutan},
                         hover_data=["Block", "Divisi"], title="CV Produksi vs Yield Gap")
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Gap = 0")
        tampil_grafik(fig)

    st.markdown("#### Distribusi fitur per segmen")
    judul = {"Yield_Aktual": "Yield Aktual (ton/ha)", "CV_Produksi": "CV Produksi (%)",
             "Achievement_Rate": "Achievement Rate (%)", "Yield_Gap": "Yield Gap (ton/ha)"}
    kb = st.columns(2)
    for i, f in enumerate(pl.FITUR):
        fig = px.box(dfc, x="Label_Segmen", y=f, color="Label_Segmen", color_discrete_map=warna,
                     category_orders={"Label_Segmen": urutan}, title=judul[f])
        fig.update_layout(showlegend=False, xaxis_title=None)
        with kb[i % 2]:
            tampil_grafik(fig)

# ───────────────────────── Tab 4 ─────────────────────────
with tab4:
    pct_neg = (dfc["Silhouette_Blok"] < 0).mean() * 100
    c = st.columns(3)
    c[0].metric("Silhouette Score", f"{res['silhouette']:.4f}")
    c[1].metric("Davies-Bouldin Index", f"{res['dbi']:.4f}")
    c[2].metric("Blok silhouette negatif", f"{pct_neg:.1f}%")

    s, d = res["silhouette"], res["dbi"]
    st.markdown("**Silhouette:** " + ("> 0,5 → cluster terpisah dengan baik ✅" if s >= 0.5 else
                "0,25–0,5 → struktur cluster cukup baik ⚠️" if s >= 0.25 else "< 0,25 → cluster lemah ❌"))
    st.markdown("**DBI:** " + ("≤ 1,0 → cluster rapat & terpisah baik ✅" if d <= 1.0 else
                "1,0–1,5 → cukup baik ⚠️" if d <= 1.5 else "> 1,5 → kurang terpisah ❌"))

    c1, c2 = st.columns(2)
    with c1:
        dm = res["df_metrik"]
        fig = go.Figure(go.Scatter(x=dm.index, y=dm["DBI"], mode="lines+markers",
                                   line=dict(color="#8E44AD", width=3)))
        fig.add_vline(x=k_opt, line_dash="dash", line_color=HIJAU, annotation_text=f"K={k_opt}")
        fig.update_layout(title="DBI per kandidat K (lebih rendah = lebih baik)",
                          xaxis_title="K", yaxis_title="DBI")
        tampil_grafik(fig)
    with c2:
        sd = dfc.sort_values(["Cluster", "Silhouette_Blok"]).reset_index(drop=True)
        fig = px.bar(sd, x="Silhouette_Blok", y=sd.index, color="Label_Segmen", orientation="h",
                     color_discrete_map=warna, category_orders={"Label_Segmen": urutan},
                     title="Silhouette per blok")
        fig.add_vline(x=s, line_dash="dash", line_color="red", annotation_text=f"Rata-rata {s:.3f}")
        fig.update_yaxes(showticklabels=False, title="Blok")
        tampil_grafik(fig)

# ───────────────────────── Tab 5 ─────────────────────────
with tab5:
    st.markdown("#### Hasil segmentasi per blok")
    pilih = st.multiselect("Filter segmen", urutan, default=urutan)
    kolom_out = ["Block", "Divisi", "Hectare"] + pl.FITUR + ["Cluster", "Label_Segmen"]
    df_out = dfc[dfc["Label_Segmen"].isin(pilih)][kolom_out] \
        .sort_values(["Label_Segmen", "Block"]).reset_index(drop=True)
    tampil_tabel(df_out.round(2))
    st.download_button("⬇️ Download CSV (semua blok)",
                       dfc[kolom_out].sort_values(["Label_Segmen", "Block"])
                       .to_csv(index=False).encode("utf-8"),
                       file_name="Hasil_Segmentasi_P2Estate.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("#### 🔮 Prediksi segmen untuk blok baru")
    st.caption("Masukkan 4 fitur sebuah blok; model K-Means yang sudah dilatih menentukan segmennya.")
    med = dfc[pl.FITUR].median()
    cols = st.columns(4)
    nilai = {f: col.number_input(f, value=float(round(med[f], 2)), format="%.2f")
             for col, f in zip(cols, pl.FITUR)}
    if st.button("Prediksi segmen", type="primary"):
        cid, label = pl.prediksi_blok_baru(res, nilai)
        st.success(f"Blok ini masuk **{label}** (Cluster {cid}).")
