# Protokol evaluasi Tim 04

**Status Sprint 0: draf untuk peninjauan.** Pembekuan menunggu anotasi dengar, pemeriksaan
lembar sesi, dan persetujuan pembimbing Ardika Satria di PR ke `main`. Rancangan mengikuti
`tim-04.pdf` dan `00-metode-eksperimen-lapangan.pdf`.

| Butir | Keputusan |
|---|---|
| Kasus | Klasifikasi multi-label peristiwa suara di meja |
| Data primer | Sesi 16 November 2026, `sesi-20261116-084442`. Arsip, sumber, dan SHA-256 tercatat di `data/MANIFEST.json`. |
| Urutan label | `buka_pintu`, `tutup_pintu`, `ketuk_meja`, `letakkan_gelas`, `kipas`, `pompa`, `buzzer`. |
| Masukan model | Audio mono 16 kHz, log-mel 64 pita × 98 frame per jendela 1 detik. |
| Split beku yang diusulkan | Latih ulangan 3 dan 5, validasi 4 dan 6, uji 1 dan 2. Pengacakan enam ID ulangan memakai `src.common.split.indeks` dan seed 2026. Dua grup pertama hasil permutasi dialokasikan ke uji, dua berikutnya ke validasi, dan sisanya ke latih. |
| Ukuran draf | **184 / 209 / 194** jendela latih / validasi / uji dari 199 kondisi. Ukuran akhir mengikuti keputusan anotasi. |
| Metrik utama | F1 per kelas. Laporkan jumlah label diketahui dan contoh positif per kelas. Macro-F1 merata-ratakan kelas yang punya contoh positif. `acc` adalah proporsi label biner benar di antara label diketahui. |
| Pembanding | MFCC 13 koefisien × statistik rerata dan simpangan baku waktu. Random Forest per label, 100 pohon, minimum 2 sampel per daun, seed 2026. |
| Sidik data draf | `4974a01ef218b520299f0a657d230eef` dari hasil baseline. Hitung kembali setelah anotasi final. |
| Anggaran komputasi | Maksimal 20 menit pelatihan per run di CPU laptop. Catat perangkat dan durasi. Untuk konfigurasi model akhir laporkan rerata ± simpangan baku dari tiga seed. Catat ukuran ONNX dan latensi di laptop meja. |
| Evaluasi uji | Sekali di sprint terakhir oleh Penjaga Data. Tetapkan konfigurasi dan ambang keputusannya dari validasi. |

## Ekstraksi dan mutu sesi

- `label.csv` memuat 240 kondisi dan 40 kombinasi aksi × alat. Setiap kombinasi muncul enam kali.
  `audio_index.csv` dan header WAV mencatat 19 potongan mono PCM 16-bit pada 48 kHz.
  Rekamannya dimulai pukul 08.44 waktu lokal dan log berakhir dengan `Selesai`.
- Cocokkan waktu `t_mulai_host` kondisi dan indeks audio. Buang detik pertama. Ambil tiga calon
  jendela utuh `[1,2)`, `[2,3)`, dan `[3,4)` relatif terhadap awal kondisi. Jendela yang
  melintasi batas WAV dikeluarkan untuk menghindari celah/tumpang tindih akibat segmentasi.
  Turunkan laju dari 48 kHz ke 16 kHz memakai resampling polifasa.
- Log perekam menandai gangguan sambungan ESP32 yang beririsan dengan kondisi 2, 3, 4, 92, 103,
  104, dan 210 serta antrean perintah penuh pada kondisi 176. Kondisi tersebut dikeluarkan.
  Untuk kondisi lain, perintah terakhir untuk masing-masing alat dalam rentang setengah detik
  sebelum hingga detik pertama kondisi harus ada dan cocok dengan `label.csv`. Secara keseluruhan
  41 kondisi dikeluarkan lewat dua pemeriksaan ini. Dari 597 calon jendela yang tersisa, 10
  melintasi batas WAV. Kecocokan perintah ini adalah pemeriksaan setelan awal, dan kualitas
  akustiknya masih perlu ditinjau dari rekaman bersama lembar sesi.
  Nomor kondisi dan jendela yang dikeluarkan ada di [catatan sesi](../../data/CATATAN-SESI.md).
- Terapkan Hann 25 ms, langkah 10 ms, FFT 512, dan filter mel segitiga HTK dari 0 hingga 8 kHz.
  Ambil log natural dari daya mel dengan batas bawah `1e-10`. Hitung rerata dan simpangan
  baku per pita hanya dari latih, lalu terapkan parameter yang sama pada validasi dan uji.
  Baseline mengambil MFCC dari log-mel sebelum normalisasi.

## Anotasi aksi

`digital.csv` tidak berisi kejadian dan kolom status pintu di `label.csv` kosong. `aksi`
mencatat isyarat untuk operator. Karena aksi dijadwalkan di awal kondisi, bunyinya bisa jatuh
pada detik transisi yang dibuang. Label aksi `-1` menandai status belum diketahui. Saat menghitung
loss atau metrik, abaikan sel `-1`. `diam` berarti tidak ada aksi setelah dikonfirmasi lewat dengar.

```bash
python -m src.protokol.anotasi
# Khusus Penjaga Data, untuk mendengar jendela uji:
python -m src.protokol.anotasi --split uji --out data/audit-dengar-uji
```

Gunakan ID jendela di `indeks.json` dan tulis hasil dengar di `anotasi_aksi.json`.
Gunakan daftar label untuk aksi yang terdengar, `[]` untuk jendela yang sudah dipastikan
tanpa bunyi aksi, atau `"keluarkan"` untuk jendela yang diketahui terganggu. ID yang belum
diputuskan boleh dibiarkan tanpa entri. Bila lembar sesi menandai gangguan satu kondisi,
keluarkan semua jendelanya dan dokumentasikan alasannya dalam peninjauan PR.

Sebelum protokol dibekukan, verifikasi bahwa setiap label memiliki contoh positif dan negatif
di tiap split. Jika bunyi aksi banyak jatuh di detik pertama, catat hasil dengar dan minta
keputusan pembimbing sebelum mengubah aturan potong atau cakupan evaluasi.

## Reproduksi baseline

```bash
python -m src.protokol.baseline --seed 2026
```

Hasil validasi di `results/baseline/metrics.json` mencakup F1 kipas, pompa, dan buzzer.
Cakupan aksi masih nol karena belum ada anotasi dengar yang terkonfirmasi. Model Sprint 1
menulis matriks prediksi biner dengan urutan label di atas. Ambang ditetapkan dari validasi.
`src.protokol.evaluate` menulis metrik dan sidik data dalam skema Workbench. Sidik ini dihitung
dari arsip, tabel anotasi, dan parameter split. Jalankan baseline kembali setelah tabel anotasi final.
