"""Jalankan baseline multi-label Tim 04: MFCC + Random Forest per label."""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.fft import dct
from sklearn.ensemble import RandomForestClassifier

from src.common.manifest import AKAR
from src.protokol import data, evaluate


def fitur(log_mel):
    mfcc = dct(log_mel, type=2, norm="ortho", axis=1)[:, :13]
    return np.concatenate((mfcc.mean(axis=2), mfcc.std(axis=2)), axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    Xtr, ytr, Xva, _, _, _ = data.muat(normalisasi=False, sertakan_uji=False)
    Xtr, Xva = fitur(Xtr), fitur(Xva)
    pred = np.zeros((len(Xva), len(data.LABEL)), dtype=np.int8)
    t0 = time.perf_counter()
    for i in range(len(data.LABEL)):
        diketahui = ytr[:, i] != data.ANOTASI_TIDAK_PASTI
        if not diketahui.any():
            continue
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=a.seed)
        model.fit(Xtr[diketahui], ytr[diketahui, i])
        pred[:, i] = model.predict(Xva)
    durasi = time.perf_counter() - t0
    out = AKAR / "results" / "baseline" / "pred_val.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred, allow_pickle=False)
    evaluate.simpan("baseline", "mfcc_rf", a.seed, str(out), durasi, "cpu")


if __name__ == "__main__":
    main()
