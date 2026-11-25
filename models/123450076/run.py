"""Jalur TL (transfer learning) (contoh kunci). Arsitektur dan hiperparameter di model.py folder ini.

    run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

try:
    from src.tim import latih  # noqa: E402
except ImportError:
    # Modul pembantu mandiri jika src.tim belum disediakan secara global
    class _LatihHelper:
        @staticmethod
        def atur_seed(seed: int) -> None:
            import random
            import torch

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        @staticmethod
        def prediksi(model, X: np.ndarray) -> np.ndarray:
            import torch

            model.eval()
            with torch.no_grad():
                xt = torch.tensor(X, dtype=torch.float32)
                return model(xt).cpu().numpy()

        @staticmethod
        def jumlah_parameter(model) -> int:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        @staticmethod
        def latensi_ms(model, X: np.ndarray) -> float:
            t0 = time.perf_counter()
            _ = _LatihHelper.prediksi(model, X[:16])
            return float((time.perf_counter() - t0) / 16.0 * 1000)

        @staticmethod
        def tulis_hasil(
            folder: str,
            jalur: str,
            seed: int,
            pred_val: np.ndarray,
            pred_test: np.ndarray | None,
            durasi: float,
            perangkat: str,
            tambahan: dict | None = None,
        ) -> None:
            import subprocess

            out = AKAR / "results" / folder
            out.mkdir(parents=True, exist_ok=True)
            pred_val_path = out / "pred_val.npy"
            np.save(pred_val_path, np.asarray(pred_val))

            perintah = [
                sys.executable,
                "-m",
                "src.protokol.evaluate",
                "--nim",
                folder,
                "--model",
                jalur,
                "--seed",
                str(seed),
                "--pred-val",
                str(pred_val_path),
                "--durasi",
                f"{durasi:.1f}",
                "--perangkat",
                perangkat,
            ]
            if pred_test is not None:
                pred_test_path = out / "pred_test.npy"
                np.save(pred_test_path, np.asarray(pred_test))
                perintah += ["--pred-test", str(pred_test_path)]

            subprocess.run(perintah, check=True, cwd=AKAR)

    latih = _LatihHelper()

FOLDER = Path(__file__).resolve().parent.name
JALUR = "tl"


def muat_model():
    spec = importlib.util.spec_from_file_location("model_jalur", Path(__file__).with_name("model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    latih.atur_seed(a.seed)
    t0 = time.time()
    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    # Penyesuaian tugas multi-label Tim 04: 7 keluaran dan dekode biner berbasis ambang
    kodek = getattr(data, "KODEK", None)  # tugas multi-label (lihat src/protokol/kodek.py)
    keluaran = kodek.n_keluaran if kodek else len(data.LABEL)

    jalur = muat_model()
    model, info = jalur.jalankan(
        Xtr,
        ytr,
        Xva,
        yva,
        Xte,
        seed=a.seed,
        latih=latih,
        cetak=FOLDER,
        keluaran=keluaran,
        kodek=kodek,
    )

    def dekode_multilabel(z: np.ndarray) -> np.ndarray:
        ambang = getattr(model, "ambang", 0.0)
        p = (z >= ambang).astype(np.int8)
        p[:, :4] = 0  # label aksi belum teranotasi
        return p

    dekode = kodek.dekode if kodek else dekode_multilabel

    logit_val = latih.prediksi(model, Xva)
    pred_val = dekode(logit_val)
    pred_test = dekode(latih.prediksi(model, Xte)) if a.final and Xte is not None else None
    durasi = time.time() - t0

    metrik_tambahan = data.metrik_tim(yva, pred_val, logit=logit_val) if hasattr(data, "metrik_tim") else {}
    tambahan = {
        "jalur": JALUR,
        "seed": a.seed,
        "hiper": jalur.HIPER,
        "parameter": latih.jumlah_parameter(model),
        "latensi_ms": latih.latensi_ms(model, Xva),
        **metrik_tambahan,
        **info,
    }
    latih.tulis_hasil(FOLDER, JALUR, a.seed, pred_val, pred_test, durasi, a.perangkat, tambahan)
    np.save(AKAR / "results" / FOLDER / "logit_val.npy", logit_val)  # untuk analisis kesalahan (tidak di-commit)


if __name__ == "__main__":
    main()
