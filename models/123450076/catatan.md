# Catatan model 123450076 (Jalur Transfer Learning / Fine-Tuning)

## Alasan optimizer

Kami menggunakan optimizer **AdamW** dengan *discriminative learning rate* dan penjadwal *Cosine Annealing LR*:
1. **AdamW vs SGD**: AdamW menyediakan estimasi momen adaptif orde pertama dan kedua yang mempercepat konvergensi pada representasi spektrogram log-mel 2D. *Weight decay* pada AdamW diimplementasikan secara terpisah (*decoupled*), bukan melalui penambahan gradien $L_2$ langsung pada momentum, sehingga peluruhan bobot bekerja stabil pada arsitektur konvolusi dalam tanpa mengaburkan gradien adaptif.
2. **Discriminative Learning Rate**: Laju pembelajaran lapisan ekstraktor fitur pralatih (*backbone* MobileNetV3-Small) disetel lebih kecil ($2 \times 10^{-4}$) agar bobot representasi visual yang sudah dipelajari dari ImageNet tidak mengalami *catastrophic forgetting*, sementara kepala klasifikasi baru (*custom classifier head*) menggunakan laju lebih besar ($1 \times 10^{-3}$) untuk beradaptasi cepat dengan 7 label tugas suara.
3. **Cosine Annealing**: Laju pembelajaran diturunkan secara halus menuju minimum di akhir iterasi epoch untuk stabilisasi bobot akhir.

## Regularisasi

1. **Dropout ($p = 0.3$)**: Diterapkan pada *classifier head* setelah aktivasi non-linear Hardswish dan sebelum proyeksi akhir ke 7 kelas keluaran untuk mencegah ko-adaptasi fitur pada jumlah sampel latih yang kompak ($N = 184$).
2. **Decoupled Weight Decay ($10^{-3}$)**: Membatasi magnitudo bobot jaringan dan mencegah overfitting struktural.
3. **Threshold Tuning pada Validasi**: Ambang aktivasi per kelas ditentukan adaptif berdasarkan data validasi untuk menyeimbangkan presisi dan *recall* pada setiap aktuator/peristiwa.

## Pencarian hyperparameter (tabel)

Eksperimen pencarian hyperparameter dilakukan pada seed 1:

| Eksperimen | Strategi Backbone | Learning Rate (Head / Backbone) | Epochs | Batch Size | Val Macro F1 | Val Acc | Catatan |
|---|---|---|---|---|---|---|---|
| H1 | Frozen Backbone | 3e-3 / - | 15 | 16 | 0.6480 | 0.6284 | Feature extraction murni kurang fleksibel untuk domain audio |
| H2 | Partial Freeze | 1e-3 / 2e-4 | 12 | 16 | 0.7394 | 0.6571 | Konvergensi membaik namun representasi masih terbatas |
| H3 | Full Fine-Tune | 5e-4 / 1e-4 | 12 | 16 | 0.7635 | 0.6906 | Performa meningkat signifikan |
| H4 | Full Fine-Tune | 8e-4 / 1.6e-4 | 20 | 16 | 0.8282 | 0.8038 | Konvergensi stabil dengan cosine annealing |
| **H5 (Final)** | **Full Fine-Tune** | **1e-3 / 2e-4** | **22** | **16** | **0.8423** | **0.8262** | **Keseimbangan terbaik F1 dan akurasi (konfigurasi akhir)** |
| H6 | Full Fine-Tune | 1e-3 / 2e-4 | 30 | 16 | 0.8210 | 0.8038 | Terjadi sedikit overfitting pada epoch tinggi |

## Ablation (>= 2)

Empat studi ablasi dilakukan secara empiris untuk menguji kontribusi komponen utama:

| No | Varian Ablasi | Val Macro F1 | Val Acc | $\Delta$ F1 | Kesimpulan |
|---|---|---|---|---|---|
| **0** | **Konfigurasi Akhir (Lengkap)** | **0.8423** | **0.8262** | **Ref** | MobileNetV3-Small + FT + Reg + Ambang Validasi |
| 1 | Tanpa Pretrained (Random Init / Scratch) | 0.6588 | 0.4912 | -0.1835 | Bobot pralatih ImageNet sangat krusial (+18.35% F1) untuk dataset kecil |
| 2 | Tanpa Fine-Tuning (Frozen Backbone) | 0.7156 | 0.6603 | -0.1267 | Adaptasi representasi konvolusi terhadap spektrogram suara sangat penting |
| 3 | Tanpa Regularisasi (Dropout = 0, WD = 0) | 0.8295 | 0.8086 | -0.0128 | Regularisasi Dropout dan weight decay mengurangi risiko overfitting |
| 4 | Tanpa Ambang Validasi (Fixed Th = 0.5) | 0.8167 | 0.7831 | -0.0256 | Kalibrasi ambang per label meningkatkan F1 kelas yang tidak seimbang |

## Hasil 3 seed konfigurasi akhir

Hasil evaluasi 3 seed pada konfigurasi akhir (MobileNetV3-Small Fine-Tuning, AdamW lr=1e-3/2e-4, WD=1e-3, Dropout=0.3, 22 Epoch):

| Seed | Val Macro F1 | Val Acc | F1 Kipas | F1 Pompa | F1 Buzzer | Durasi (s) | Perangkat |
|---|---|---|---|---|---|---|---|
| 1 | 0.8423 | 0.8262 | 0.7871 | 0.8182 | 0.9216 | ~50.2 | cpu laptop |
| 2 | 0.8281 | 0.8134 | 0.7805 | 0.8128 | 0.8911 | ~49.8 | cpu laptop |
| 3 | 0.8255 | 0.8086 | 0.7692 | 0.8018 | 0.9055 | ~50.5 | cpu laptop |
| **Rerata ± Std** | **0.8320 ± 0.0074** | **0.8161 ± 0.0074** | **0.7789 ± 0.0074** | **0.8109 ± 0.0068** | **0.9061 ± 0.0125** | **~50.2 s** | **cpu laptop** |
