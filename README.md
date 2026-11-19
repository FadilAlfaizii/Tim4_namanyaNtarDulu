# Capstone Deep Learning — Tim 04

Tim 04 mengerjakan **klasifikasi multi-label peristiwa suara di meja** untuk Capstone Deep
Learning Program Studi Sains Data ITERA. Empat jalur model memakai rekaman meja dan protokol
evaluasi yang sama. Setiap anggota bertanggung jawab atas satu keluarga model dan satu folder.

## Struktur

```text
src/common/        milik starter — jangan diubah: evaluate (penilai), split (split beku), manifest (data_hash), baseline
src/protokol/      milik tim, DIBEKUKAN setelah Sprint 0 dengan persetujuan pembimbing: data.py (pemuat kasus tim), PROTOKOL.md
data/              dataset kasus tim (tidak di-commit; lihat data/MANIFEST.json)
models/<nim>/      milik satu anggota — kode model, run.py
results/<nim>/     berkas hasil anggota: pred_val.npy, pred_test.npy, metrics.json
reports/individu/<nim>.md   laporan perorangan (≤ 1200 kata)
reports/tim.md     laporan tim (6 bagian)
AI_LOG.md          catatan pemakaian AI tim
run_all.py         satu perintah: jalankan semua model dan rakit tabel komparasi
```

## Sprint 0: membekukan protokol tim

1. Letakkan salinan arsip sesi Tim 04 di `data/tim04_sesi-20261116-084442.tar.gz`. Sumber
   dan SHA-256 ada di `data/MANIFEST.json`. Arsip lokal diabaikan Git.
2. Gunakan `src/protokol/data.py`. Fungsi `muat()` mengembalikan `X_train, y_train, X_val,
   y_val, X_test, y_test`. Split memakai ID ulangan dengan seed 2026.
3. Ikuti [protokol Tim 04](src/protokol/PROTOKOL.md) untuk anotasi dengar, aturan ekstraksi,
   urutan label, dan penilaian multi-label. Lengkapi anotasi dan verifikasi lembar sesi.
4. Jalankan baseline MFCC + Random Forest untuk memeriksa jalur evaluasi:

   ```bash
   pip install -r requirements.txt
   python -m src.protokol.baseline --seed 2026
   ```

   Hasil validasi ada di `results/baseline/metrics.json`. Selama anotasi aksi belum lengkap,
   cakupan label dilaporkan bersama skor.

5. Penjaga Data & Evaluasi meminta persetujuan pembimbing lewat *pull request* ke `main`.
   Setelah digabung, perubahan `src/protokol/` memerlukan *pull request* yang disetujui pembimbing.

## Sprint 1 dan seterusnya: model tiap anggota

1. Salin `models/TEMPLATE/` menjadi `models/<nim>/`, isi `run.py` sesuai `models/README.md`.
2. Jalankan tiga seed pada konfigurasi akhir:

   ```bash
   python models/<nim>/run.py --seed 1 && python models/<nim>/run.py --seed 2 && python models/<nim>/run.py --seed 3
   ```

   Setiap run menulis `results/<nim>/metrics.json` lewat `src.protokol.evaluate`. Local Runner
   memeriksanya (skema `metrik/seed/data_hash`, isi sama dengan HEAD) saat *Periksa artefak*
   ditekan di aplikasi DSWorkbench.
3. Perintah reproduksi tim (dinilai pada sprint terakhir): `python run_all.py`.

## Yang dibakukan untuk Tim 04

- Split beku per ulangan dengan seed 2026 (`src/common/split.py`). Data uji
  hanya dievaluasi **sekali** oleh Penjaga Data pada sprint terakhir (`--final`).
- Metrik utama adalah F1 per kelas. Hasil konfigurasi akhir dilaporkan sebagai rerata ±
  simpangan baku dari 3 seed.
- Anggaran komputasi ≤ 20 menit pelatihan per run di CPU laptop (atau sesuai yang disetujui
  pembimbing untuk kasus tim); catat `--durasi` dan `--perangkat`.
- `data_hash` dihitung dari berkas di `data/MANIFEST.json` dan split; jangan diisi manual.

## Jalur model

| Jalur | Keluarga | Modul |
|---|---|---|
| `cnn` | CNN (1D/2D sesuai data) | 05 |
| `tl` | Transfer learning / fine-tuning model pralatih | 06 |
| `trf` | Transformer encoder | 09 |
| `ae` | Autoencoder → representasi untuk klasifikasi/regresi/anomali | — |

Pembimbing boleh menyesuaikan pilihan jalur dengan sifat data kasus tim. Syarat semua model:
optimizer dengan alasan tertulis, ≥ 1 regularisasi, tuning hyperparameter dengan catatan
pencarian, ≥ 2 ablation, 3 seed pada konfigurasi akhir.

Lisensi: MIT. Repo ini masuk Galeri Capstone sebagai portofolio setelah semester berakhir.
