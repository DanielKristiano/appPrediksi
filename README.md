# Deployment Streamlit — Segmentasi Blok P2 Estate

## Struktur
```
app.py            # tampilan Streamlit
.streamlit/config.toml  # tema warna
pipeline.py       # logika dari notebook (cleaning, fitur, K-Means, evaluasi)
requirements.txt
Dataset Primer.xlsx   # (opsional) taruh di sini agar terbaca otomatis
```

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy gratis (Streamlit Community Cloud)
1. Upload semua file ke repo GitHub.
2. Buka https://share.streamlit.io → New app → pilih repo, branch, file `app.py`.
3. Deploy. Kalau dataset bersifat rahasia, JANGAN commit `Dataset Primer.xlsx`;
   cukup upload lewat sidebar saat app dibuka.
