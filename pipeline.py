"""
Pipeline segmentasi blok kebun sawit P2 Estate (K-Means).
Logika disalin dari notebook skripsi (Fase 3 - 5) supaya hasil app == hasil notebook.
File ini TIDAK bergantung pada Streamlit, jadi bisa diuji terpisah.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA

KOLOM_WAJIB = ["Block", "Divisi", "Month", "Hectare Mature",
               "Actual FFB Production CM", "Budget FFB Production CM"]
KOLOM_NUMERIK = ["Actual FFB Production CM", "Budget FFB Production CM", "Hectare Mature"]
FITUR_A = ["Yield_Aktual", "CV_Produksi"]
FITUR_B = ["Achievement_Rate", "Yield_Gap"]
FITUR = FITUR_A + FITUR_B


# ───────────────────────── Fase 3: Data Preparation ─────────────────────────
def cek_kolom(df_raw):
    df_raw = df_raw.copy()
    df_raw.columns = df_raw.columns.astype(str).str.strip()
    hilang = [c for c in KOLOM_WAJIB if c not in df_raw.columns]
    return df_raw, hilang


def bersihkan_data(df_raw):
    """3.1 Konversi tipe, buang baris tanpa Block, imputasi median per Divisi."""
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    for col in KOLOM_NUMERIK:
        df[col] = (df[col].astype(str).str.replace(",", "", regex=False)
                   .str.strip().replace("nan", np.nan).astype(float))
    df = df.dropna(subset=["Block"]).reset_index(drop=True)
    for col in KOLOM_NUMERIK:
        if df[col].isnull().sum() == 0:
            continue
        med_div = df.groupby("Divisi")[col].transform("median")
        df[col] = df[col].fillna(med_div).fillna(df[col].median())
    return df


def buat_fitur_blok(df):
    """3.2 Agregasi panel -> 1 baris per blok + 4 fitur."""
    dfb = df.groupby("Block").agg(
        Divisi=("Divisi", "first"),
        Hectare=("Hectare Mature", "mean"),
        Total_Actual=("Actual FFB Production CM", "sum"),
        Total_Budget=("Budget FFB Production CM", "sum"),
        Mean_Actual=("Actual FFB Production CM", "mean"),
        Std_Actual=("Actual FFB Production CM", "std"),
    ).reset_index()
    ha = dfb["Hectare"].replace(0, np.nan)
    dfb["Yield_Aktual"] = dfb["Total_Actual"] / ha / 1000
    dfb["CV_Produksi"] = dfb["Std_Actual"] / dfb["Mean_Actual"].replace(0, np.nan) * 100
    dfb["Achievement_Rate"] = dfb["Total_Actual"] / dfb["Total_Budget"].replace(0, np.nan) * 100
    dfb["Yield_Budget"] = dfb["Total_Budget"] / ha / 1000
    dfb["Yield_Gap"] = dfb["Yield_Budget"] - dfb["Yield_Aktual"]
    return dfb


# ───────────────────────── Fase 4: Modeling ─────────────────────────
def detect_elbow(k_values, wcss):
    k = np.asarray(k_values, float)
    w = np.asarray(wcss, float)
    kn = (k - k.min()) / (k.max() - k.min())
    wn = (w - w.min()) / (w.max() - w.min())
    x1, y1, x2, y2 = kn[0], wn[0], kn[-1], wn[-1]
    num = np.abs((y2 - y1) * kn - (x2 - x1) * wn + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return int(k[int(np.argmax(num / den))])


def pilih_k(df_metrik):
    elbow_k = detect_elbow(df_metrik.index.values, df_metrik["WCSS"].values)
    sil_k = int(df_metrik["Silhouette"].idxmax())
    s_el, s_si = df_metrik.loc[elbow_k, "Silhouette"], df_metrik.loc[sil_k, "Silhouette"]
    if elbow_k == sil_k:
        k_opt = elbow_k
        alasan = f"Elbow dan Silhouette sepakat pada K={k_opt}."
    elif s_si >= s_el:
        k_opt = sil_k
        alasan = (f"Elbow menunjuk K={elbow_k} (Silhouette {s_el:.4f}), Silhouette menunjuk "
                  f"K={sil_k} (skor lebih tinggi, {s_si:.4f}) → dipakai K={k_opt}.")
    else:
        k_opt = elbow_k
        alasan = (f"Elbow menunjuk K={elbow_k} dengan Silhouette {s_el:.4f}, lebih tinggi dari "
                  f"K={sil_k} ({s_si:.4f}) → dipakai K={k_opt}.")
    return elbow_k, sil_k, k_opt, alasan


def jalankan_segmentasi(df_block, k_min=2, k_max=8, k_manual=None,
                        n_init=20, random_state=42, clip_q=(0.005, 0.995)):
    """4.1 - 4.4 + 5.2: preprocessing, scan K, pilih K, label segmen, evaluasi."""
    dfc = df_block[["Block", "Divisi", "Hectare"] + FITUR].copy()
    for c in FITUR:                                   # jaga-jaga NaN
        dfc[c] = dfc[c].fillna(dfc[c].median())

    batas = {}
    for c in FITUR:                                   # kliping outlier 0.5% - 99.5%
        lo, hi = dfc[c].quantile(clip_q[0]), dfc[c].quantile(clip_q[1])
        batas[c] = (float(lo), float(hi))
        dfc[c] = dfc[c].clip(lo, hi)

    scaler = StandardScaler()
    X = scaler.fit_transform(dfc[FITUR])

    hasil, models = [], {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                    max_iter=500, random_state=random_state)
        lab = km.fit_predict(X)
        hasil.append({"K": k, "WCSS": km.inertia_,
                      "Silhouette": silhouette_score(X, lab),
                      "DBI": davies_bouldin_score(X, lab)})   # DBI hanya utk evaluasi
        models[k] = km
    df_metrik = pd.DataFrame(hasil).set_index("K")

    elbow_k, sil_k, k_auto, alasan = pilih_k(df_metrik)
    if k_manual is not None and k_manual in models:
        k_opt, alasan = k_manual, f"K dipilih manual oleh pengguna (K={k_manual})."
    else:
        k_opt = k_auto
    km_opt = models[k_opt]
    dfc["Cluster"] = km_opt.labels_

    # Pelabelan berdasarkan skor kinerja gabungan (sama dgn notebook 4.4.1)
    profil = dfc.groupby("Cluster")[FITUR].mean()
    if len(profil) > 1:
        pn = pd.DataFrame(MinMaxScaler().fit_transform(profil), index=profil.index, columns=FITUR)
    else:
        pn = profil * 0
    pn["Skor_Kinerja"] = pn["Yield_Aktual"] + pn["Achievement_Rate"] - pn["CV_Produksi"] - pn["Yield_Gap"]
    rank = pn["Skor_Kinerja"].rank(ascending=False, method="first").astype(int)
    label_map = {}
    for cid, r in rank.items():
        if r == 1:
            label_map[cid] = "Segmen 1: Kinerja Tinggi"
        elif r == k_opt:
            label_map[cid] = f"Segmen {k_opt}: Kinerja Rendah"
        else:
            label_map[cid] = f"Segmen {r}: Kinerja Sedang"
    dfc["Label_Segmen"] = dfc["Cluster"].map(label_map)
    urutan_label = [label_map[c] for c in sorted(label_map, key=lambda c: rank[c])]

    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)
    dfc["PCA1"], dfc["PCA2"] = Xp[:, 0], Xp[:, 1]
    cent_pca = pca.transform(km_opt.cluster_centers_)
    dfc["Silhouette_Blok"] = silhouette_samples(X, dfc["Cluster"].values)

    return {
        "df_cluster": dfc, "X": X, "df_metrik": df_metrik,
        "elbow_k": elbow_k, "sil_k": sil_k, "k_opt": k_opt, "alasan_k": alasan,
        "silhouette": float(df_metrik.loc[k_opt, "Silhouette"]),
        "dbi": float(df_metrik.loc[k_opt, "DBI"]),
        "label_map": label_map, "urutan_label": urutan_label,
        "var_exp": pca.explained_variance_ratio_ * 100, "cent_pca": cent_pca,
        "scaler": scaler, "km": km_opt, "batas": batas,
    }


def prediksi_blok_baru(res, nilai):
    """Prediksi segmen untuk 1 blok baru dari 4 nilai fitur (dict)."""
    row = pd.DataFrame([{c: float(np.clip(nilai[c], *res["batas"][c])) for c in FITUR}])
    cid = int(res["km"].predict(res["scaler"].transform(row))[0])
    return cid, res["label_map"][cid]
