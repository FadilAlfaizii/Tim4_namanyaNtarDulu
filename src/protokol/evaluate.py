"""Penilai multi-label Tim 04 dengan skema hasil Workbench."""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.common.manifest import AKAR
from src.protokol import data


def metrik(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if pred.shape != y.shape or not np.isin(pred, (0, 1)).all():
        raise ValueError(f"Prediksi harus berupa matriks biner berukuran {y.shape}.")
    hasil = {}
    f1 = []
    for k, nama in enumerate(data.LABEL):
        dikenal = y[:, k] != data.ANOTASI_TIDAK_PASTI
        a, p = y[dikenal, k], pred[dikenal, k]
        positif = int(a.sum())
        hasil[f"n_{nama}"] = len(a)
        hasil[f"positif_{nama}"] = positif
        if positif:
            nilai = 2 * np.sum((a == 1) & (p == 1)) / (positif + p.sum())
            hasil[f"f1_{nama}"] = float(nilai)
            f1.append(nilai)
    if f1:
        hasil["macro_f1"] = float(np.mean(f1))
    dikenal = y != data.ANOTASI_TIDAK_PASTI
    if dikenal.any():
        hasil["acc"] = float((pred[dikenal] == y[dikenal]).mean())
    return hasil


def nilai(pred_val: str, pred_test: str | None = None):
    target = [y[pakai] for _, y, pakai in data.indeks_data(sertakan_uji=pred_test is not None)]
    _, y_val = target[:2]
    hasil = {f"val_{k}": v for k, v in metrik(y_val, np.load(pred_val, allow_pickle=False)).items()}
    ukuran = {"n_val": len(y_val), "n_test": 0}
    if pred_test is not None:
        for nama, y in zip(("latih", "validasi", "uji"), target):
            if (y < 0).any() or not ((y == 1).any(axis=0) & (y == 0).any(axis=0)).all():
                raise ValueError(f"Evaluasi final memerlukan anotasi lengkap dan contoh positif serta negatif "
                                 f"untuk tiap label di split {nama}.")
        y_test = target[2]
        hasil.update({f"test_{k}": v for k, v in metrik(y_test, np.load(pred_test, allow_pickle=False)).items()})
        ukuran["n_test"] = len(y_test)
    return hasil, ukuran


def simpan(nim, model, seed, pred_val, durasi, perangkat, pred_test=None):
    hasil, ukuran = nilai(pred_val, pred_test)
    keluaran = {
        "metrik": {k: round(v, 6) for k, v in hasil.items()},
        "seed": seed, "data_hash": data.data_hash(), "model": model,
        "durasi_detik": round(durasi, 3), "perangkat": perangkat, **ukuran,
    }
    out = AKAR / "results" / nim / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(keluaran, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(keluaran, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nim", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pred-val", required=True)
    ap.add_argument("--pred-test", help="Hanya untuk Penjaga Data pada sprint terakhir")
    ap.add_argument("--durasi", type=float, required=True)
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    simpan(a.nim, a.model, a.seed, a.pred_val, a.durasi, a.perangkat, a.pred_test)


if __name__ == "__main__":
    main()
