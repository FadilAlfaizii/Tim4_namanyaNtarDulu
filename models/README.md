# Kontrak `models/<nim>/run.py`

Setiap anggota memiliki satu folder `models/<nim>/` (hanya pemiliknya yang boleh mengubah).
Isinya bebas (PyTorch dianjurkan, sesuai Modul 03–09), dengan satu syarat: skrip `run.py`
yang dapat dipanggil `run_all.py` dan menulis hasil lewat penilai dosen.

```bash
python models/<nim>/run.py --seed 1            # latih + prediksi validasi
python models/<nim>/run.py --seed 1 --final    # sprint terakhir: + prediksi data uji (sekali)
```

Yang wajib dilakukan `run.py`:

1. Memuat data lewat `src.protokol.data.muat(sertakan_uji=final)`. Split sudah ditentukan per ulangan.
2. Menetapkan seed dari argumen `--seed` ke NumPy dan PyTorch (`torch.manual_seed`).
3. Menulis prediksi validasi ke `results/<nim>/pred_val.npy` sebagai matriks biner `(N, 7)`
   dengan urutan kolom `src.protokol.data.LABEL`. Dengan `--final`, tulis juga
   `results/<nim>/pred_test.npy`. Saat menghitung loss, abaikan sel target yang bernilai `-1`.
   Tetapkan ambang prediksi dari validasi.
4. Memanggil penilai multi-label Tim 04:

   ```python
   subprocess.run([sys.executable, "-m", "src.protokol.evaluate", "--nim", NIM, "--model", "cnn",
                   "--seed", str(seed), "--pred-val", "results/NIM/pred_val.npy",
                   "--durasi", f"{durasi:.1f}", "--perangkat", "cpu laptop"], check=True, cwd=AKAR)
   ```

5. Menyimpan catatan pencarian hyperparameter dan hasil ablation di `models/<nim>/catatan.md`
   (angka ringkasan 3 seed juga ditulis di laporan perorangan).

`metrics.json` yang ditulis penilai adalah **berkas hasil** yang diperiksa Local Runner
(*Periksa artefak* di aplikasi DSWorkbench). Jangan menyuntingnya dengan tangan; jalankan
ulang `run.py` bila ingin mengganti angkanya, lalu *commit*.

Folder `models/TEMPLATE/` berisi kerangka `run.py` yang sudah memenuhi kontrak ini.
