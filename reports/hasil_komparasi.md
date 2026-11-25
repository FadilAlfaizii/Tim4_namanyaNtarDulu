# Hasil Komparasi: Model Transfer Learning vs Baseline

Dokumen ini memuat hasil komparasi antara model **Transfer Learning / Fine-Tuning** (`tl` — NIM 123450076) dengan model **Baseline** non-DL (`mfcc_rf`) pada kasus klasifikasi multi-label peristiwa suara meja Tim 04.

---

## 1. Tabel Komparasi Utama

Cakupan label dan F1 per kelas tercatat di `results/<nim>/metrics.json`. Evaluasi data uji (`test_macro_f1`) hanya dievaluasi sekali pada sprint terakhir (`--final`).

| NIM | Model | seed | val_acc | val_macro_f1 | test_macro_f1 | durasi (s) | perangkat |
|---|---|---|---|---|---|---|---|
| baseline | mfcc_rf | 2026 | 0.9250 | 0.9236 | — | 1.8 | cpu |
| 123450076 | tl | 1 | 0.8262 | 0.8423 | — | 61.7 | cpu laptop |
| 123450076 | tl | 2 | 0.8134 | 0.8281 | — | 63.5 | cpu laptop |
| 123450076 | tl | 3 | 0.8086 | 0.8255 | — | 62.8 | cpu laptop |
| **123450076 (Rerata ± Std)** | **tl** | **—** | **0.8161 ± 0.0074** | **0.8320 ± 0.0074** | **—** | **62.7 ± 0.8** | **cpu laptop** |

> **Catatan Validitas**: Seluruh model diuji pada split beku yang sama (seed 2026, $N_{\text{val}} = 209$) dengan sidik data identik: `data_hash = 4974a01ef218b520299f0a657d230eef`.

---

## 2. Rincian Metrik Per Kelas (Akurasi & F1-Score)

Perbandingan performa deteksi pada masing-masing label aktuator (label aksi belum teranotasi pada Sprint 0/1 sehingga bernilai `-1`):

| Label | Metrik | Baseline (`mfcc_rf`) | TL Seed 1 | TL Seed 2 | TL Seed 3 | TL (Rerata ± Std) | Selisih (TL vs Baseline) |
|---|---|---|---|---|---|---|---|
| **Buzzer** | F1-Score | **1.0000** | 0.9216 | 0.8911 | 0.9055 | 0.9061 ± 0.0125 | -0.0939 |
| **Pompa** | F1-Score | **0.8626** | 0.8182 | 0.8128 | 0.8018 | 0.8109 ± 0.0068 | -0.0517 |
| **Kipas** | F1-Score | **0.9082** | 0.7871 | 0.7805 | 0.7692 | 0.7789 ± 0.0074 | -0.1293 |
| **Macro F1** | Mean F1 | **0.9236** | 0.8423 | 0.8281 | 0.8255 | 0.8320 ± 0.0074 | -0.0916 |
| **Akurasi** | Subset/Hamming | **0.9250** | 0.8262 | 0.8134 | 0.8086 | 0.8161 ± 0.0074 | -0.1089 |

---

## 3. Perbandingan Karakteristik Arsitektur & Rekayasa Fitur

| Dimensi | Baseline (`mfcc_rf`) | Transfer Learning (`tl` — MobileNetV3-Small) |
|---|---|---|
| **Paradigma** | Pembelajaran Mesin Klasik (Non-DL) | Deep Learning (Transfer Learning & Fine-Tuning) |
| **Representasi Masukan** | Vektor Tabular Statistik 26 dimensi (13 koefisien MFCC × rerata & std waktu) | Tensor Spektrogram Log-Mel 2D 3-Kanal ($3 \times 64 \text{ pita} \times 98 \text{ frame}$) |
| **Basis Fitur Pralatih** | Tidak ada (dihitung langsung dari sinyal audio) | Bobot awal konvolusi dari ImageNet-1K (fitur tepi, tekstur, dan motif visual) |
| **Strategi Multi-label** | *Binary Relevance* (melatih 7 model Random Forest independen per label) | *Joint End-to-End Learning* (satu jaringan saraf dengan 7 neuron keluaran logit bersama) |
| **Mekanisme Atensi** | Tidak ada | *Squeeze-and-Excitation* (SE) per kanal pada tiap blok *inverted residual* |
| **Regularisasi** | `min_samples_leaf = 2`, `n_estimators = 100` | Dropout ($p=0.3$), Decoupled Weight Decay ($10^{-3}$), Cosine Annealing LR |
| **Kalibrasi Ambang** | Ambang probabilitas standar $0.5$ | *Threshold Tuning* adaptif per kelas dari data validasi ($\tau \in [0.2, 0.8]$) |

---

## 4. Analisis Mekanistik & Perbedaan Hasil

### A. Mengapa Baseline MFCC + Random Forest Sangat Kuat pada Tahap Ini?
1. **Efisiensi Sampel (*Sample Efficiency*) pada Data Kecil**:
   Ukuran set data latih hanya **$N = 184$ jendela audio**. Baseline memadatkan spektrogram menjadi 26 fitur statistik global, menghasilkan rasio sampel-ke-fitur yang sehat ($\approx 7 : 1$). Random Forest sangat unggul dalam mempartisi ruang fitur berdimensi rendah tanpa risiko overfitting berlebih.
2. **Karakteristik Akustik Aktuator Statis**:
   Suara kipas, pompa, dan buzzer bersifat stasioner selama durasi 1 detik. Rerata dan simpangan baku MFCC sudah sangat memadai untuk memisahkan profil frekuensi motor dan nada murni buzzer, sehingga model tabular sederhana mencapai F1 makro **0.9236**.

### B. Analisis Performa Model Transfer Learning (MobileNetV3-Small)
1. **Tantangan Cross-Domain (Visual ke Akustik)**:
   MobileNetV3 dipralatih pada citra natural RGB ImageNet di mana terdapat invarian translasi spasial 2D. Pada spektrogram audio, sumbu vertikal menyatakan frekuensi absolut sedangkan sumbu horizontal menyatakan waktu, sehingga asimetri ini menuntut adaptasi filter konvolusi.
2. **Kebutuhan Fine-Tuning**:
   Berdasarkan studi ablasi di `models/123450076/catatan.md`:
   - Melatih MobileNetV3 dari awal (*scratch*) hanya menghasilkan F1 = **0.6588**.
   - Membekukan seluruh *backbone* (*frozen feature extraction*) hanya menghasilkan F1 = **0.7156**.
   - Menerapkan *discriminative fine-tuning* menaikkan F1 secara signifikan menjadi **0.8423** (+18.35% dibanding scratch). Hal ini membuktikan bahwa bobot pralatih ImageNet memberikan *prior* representasi yang sangat bernilai.
3. **Stabilitas Antar-Seed**:
   Model transfer learning menunjukkan variasi seed yang sangat rendah:
   $$\text{Macro F1} = 0.8320 \pm 0.0074 \quad (\text{rentang: } 0.8255 - 0.8423)$$
   Hal ini membuktikan bahwa optimizer AdamW dengan penjadwal Cosine Annealing berhasil membawa model ke konvergensi yang konsisten.

---

## 5. Efisiensi Komputasi & Anggaran

| Indikator | Baseline (`mfcc_rf`) | Transfer Learning (`tl`) | Batas Anggaran Tim | Status Kepatuhan |
|---|---|---|---|---|
| **Waktu Pelatihan** | ~1.8 detik | ~62.7 detik | $\le 20$ menit (1200 s) | **Sangat Patuh** ($< 6\%$ dari batas) |
| **Waktu Inferensi Val** | ~0.05 detik | ~0.35 detik | Real-time ($\le 1.0$ s) | **Patuh** |
| **Perangkat** | CPU | CPU Laptop | CPU Laptop | **Sesuai Mandat** |
| **Jumlah Parameter** | ~0.05 MB (pohon RF) | ~2.54 Juta parameter | Ringan untuk edge | **Sesuai Target** |

---

## 6. Kesimpulan & Rekomendasi

1. **Kesimpulan Komparasi**:
   Untuk deteksi aktuator stasioner (kipas, pompa, buzzer) dengan data latih terbatas ($N=184$), **Baseline MFCC + RF** unggul dalam akurasi dan kecepatan komputasi instan. Namun, **Transfer Learning MobileNetV3-Small** berhasil menunjukkan performa solid (F1 makro $0.8320$) dengan stabilitas seed yang sangat tinggi.
2. **Prospek untuk Label Aksi (Sprint Akhir)**:
   Keunggulan inheren arsitektur MobileNetV3 2D terletak pada kemampuannya mengekstrak pola transien dan resolusi temporal lokal. Saat label aksi fisik manusia (`buka_pintu`, `tutup_pintu`, `ketuk_meja`, `letakkan_gelas`) diaktifkan pada sprint akhir, fitur agregat rata-rata waktu pada baseline MFCC diprediksi akan mengalami degradasi karena kehilangan dinamika temporal, sedangkan model deep learning konvolusional 2D diperkirakan akan lebih adaptif mengenali pola onset suara transien tersebut.
