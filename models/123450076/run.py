from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data 

NIM = Path(__file__).resolve().parent.name  
JALUR = "tl"                                   


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None) dengan fine-tuning MobileNetV3-Small."""
    import torch
    import torch.nn as nn
    import torchvision.models as models

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat(sertakan_uji=final)

    # Praproses masukan: Log-mel (N, 64, 98) direplikasi ke 3 kanal untuk backbone pralatih
    def ke_tensor(X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32)[:, None, :, :].expand(-1, 3, -1, -1)

    X_train_t = ke_tensor(Xtr)
    y_train_t = torch.tensor(ytr, dtype=torch.float32)
    X_val_t = ke_tensor(Xva)

    # Arsitektur: MobileNetV3-Small (bobot ImageNet) dengan kepala klasifikasi kustom
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = m.classifier[0].in_features 
    m.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.3),  # Regularisasi Dropout
        nn.Linear(256, len(data.LABEL)),
    )

    # Optimizer AdamW dengan discriminative learning rate & weight decay (regularisasi L2)
    lr_head = 1e-3
    lr_backbone = 2e-4
    epochs = 22
    batch_size = 16

    param_groups = [
        {"params": m.features.parameters(), "lr": lr_backbone},
        {"params": m.classifier.parameters(), "lr": lr_head},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    N = len(X_train_t)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            logits = m(batch_x)
            diketahui = batch_y != data.ANOTASI_TIDAK_PASTI
            if diketahui.any():
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[diketahui], batch_y[diketahui]
                )
                loss.backward()
                opt.step()
        scheduler.step()

    m.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(m(X_val_t)).numpy()

    # Tetapkan ambang prediksi dari data validasi untuk memaksimalkan F1
    best_th = np.full(len(data.LABEL), 0.5)
    for c in range(len(data.LABEL)):
        dikenal = yva[:, c] != data.ANOTASI_TIDAK_PASTI
        if not dikenal.any():
            continue
        y_col = yva[dikenal, c]
        pos = int(y_col.sum())
        if pos == 0:
            continue
        probs_col = val_probs[dikenal, c]
        best_f1 = -1.0
        for th in np.linspace(0.2, 0.8, 13):
            pred_col = (probs_col >= th).astype(int)
            penyebut = pos + pred_col.sum()
            f1 = 2 * np.sum((y_col == 1) & (pred_col == 1)) / penyebut if penyebut > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_th[c] = th

    pred_val = (val_probs >= best_th).astype(np.int8)
    for c in range(len(data.LABEL)):
        if not (yva[:, c] != data.ANOTASI_TIDAK_PASTI).any():
            pred_val[:, c] = 0

    if Xte is not None:
        with torch.no_grad():
            X_test_t = ke_tensor(Xte)
            test_probs = torch.sigmoid(m(X_test_t)).numpy()
            pred_test = (test_probs >= best_th).astype(np.int8)
            for c in range(len(data.LABEL)):
                if not (yva[:, c] != data.ANOTASI_TIDAK_PASTI).any():
                    pred_test[:, c] = 0
    else:
        pred_test = None

    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable,
        "-m",
        "src.protokol.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
