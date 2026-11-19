# Catatan sesi Tim 04

Sesi `sesi-20261116-084442`, 16 November 2026 pukul 08.44–09.02, direkam di Meja Uji
Ruang Cerdas 5 V. Log berakhir dengan `Selesai` dan manifest perekam menandai
`simulasi: false`. Arsip dan SHA-256 `label.csv` tercatat di [MANIFEST.json](MANIFEST.json).

| Pemeriksaan | Hasil |
|---|---|
| Rencana dan rekaman | 240 kondisi, 40 kombinasi, enam ulangan per kombinasi. |
| Audio | 19 WAV mono PCM 16-bit 48 kHz. Indeks dan panjang WAV cocok. Semua jendela terpilih memiliki sinyal nonnol. |
| Sensor pintu | `digital.csv` hanya berisi header. Status awal dan akhir pintu kosong di `label.csv`. |
| Kelayakan awal | 199 kondisi dan 587 jendela tersedia untuk ditinjau. |
| Pemeriksaan dengar dan lembar sesi | Belum ada catatan dengar terkonfirmasi atau salinan lembar kejadian pada arsip ini. |

Kondisi yang dikeluarkan karena gangguan ESP32 di log perekam

- Putus sambungan pada rentang kondisi 2, 3, 4, 92, 103, 104, 210.
- Antrean perintah penuh pada kondisi 176.

Kondisi lain yang dikeluarkan karena setelan awal alat tidak lengkap atau tidak cocok di
`aktuator.csv`

- 1, 62, 90, 91, 93–102, 108–114, 186, 193, 194, 197, 201, 205, 206, 208,
  209, 214, 229, 240.

Jendela yang dikeluarkan karena melintasi batas potongan WAV, dalam format
`nomor kondisi:detik awal`

- `13:3`, `40:1`, `53:2`, `80:1`, `120:1`, `133:2`, `160:1`,
  `173:2`, `200:1`, `213:2`.

Aturan ekstraksi, format anotasi dengar, dan pemeriksaan sebelum pembekuan ada di
[protokol evaluasi](../src/protokol/PROTOKOL.md).
