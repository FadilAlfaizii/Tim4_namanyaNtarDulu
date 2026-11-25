# Hasil Komparasi

Dokumen ini memuat hasil komparasi antara model **Deep Learning**  dengan model **Non-Deep Learning** (Baseline `mfcc_rf`) pada kasus klasifikasi multi-label peristiwa suara meja Tim 04.

---

Cakupan label dan F1 per kelas tercatat di `results/<nim>/metrics.json`. Evaluasi data uji (`test_acc`) hanya dievaluasi sekali pada sprint terakhir (`--final`).

| NIM | Model | seed | val_acc | val_macro_f1 | test_acc | durasi (s) | perangkat |
|---|---|---|---|---|---|---|---|
| baseline | mfcc_rf | 2026 | 0.9250 | 0.9236 | — | 1.8 | cpu |
| 123450076 | tl | 1 | 0.8262 | 0.8423 | — | 68.6 | cpu laptop |