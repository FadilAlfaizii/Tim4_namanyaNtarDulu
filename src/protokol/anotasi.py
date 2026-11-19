"""Ekspor WAV dan indeks jendela untuk anotasi dengar Tim 04."""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from src.common.manifest import AKAR
from src.protokol import data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("latih", "validasi", "uji"), nargs="+",
                    default=("latih", "validasi"))
    ap.add_argument("--out", type=Path, default=AKAR / "data" / "audit-dengar")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    grup = dict(zip(("latih", "validasi", "uji"), data.grup_split()))
    partisi = {u: nama for nama in a.split for u in grup[nama]}
    indeks = {}
    for j in data.ekstrak():
        if j.ulangan not in partisi:
            continue
        nama = f"{j.kondisi:03d}-{j.detik}s.wav"
        pcm = np.rint(j.audio.astype(np.float64) * 32768).clip(-32768, 32767).astype("<i2")
        with wave.open(str(a.out / nama), "wb") as f:
            f.setparams((1, 2, data.RATE, 0, "NONE", "not compressed"))
            f.writeframes(pcm.tobytes())
        indeks[j.id] = {
            "klip": nama, "split": partisi[j.ulangan], "ulangan": j.ulangan,
            "aksi_isyarat": j.aksi_isyarat, "mulai_host": j.mulai_host,
            "wav_asli": j.berkas, "sampel_awal_asli": j.sampel_awal,
        }
    (a.out / "indeks.json").write_text(json.dumps(indeks, indent=2) + "\n", encoding="utf-8")
    print(f"{len(indeks)} klip untuk anotasi di {a.out}")


if __name__ == "__main__":
    main()
