"""Arsitektur dan hiperparameter Jalur TL (Transfer Learning MobileNetV3-Small)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from src.protokol import data

HIPER = {
    "arsitektur": "MobileNetV3-Small",
    "bobot_awal": "ImageNet-1K (MobileNet_V3_Small_Weights.DEFAULT)",
    "lr_backbone": 2e-4,
    "lr_head": 1e-3,
    "weight_decay": 1e-3,
    "epochs": 22,
    "batch_size": 16,
    "dropout": 0.3,
    "scheduler": "CosineAnnealingLR",
    "loss": "BCEWithLogitsLoss (masking target -1)",
}


class ModelTL(nn.Module):
    """MobileNetV3-Small yang disesuaikan untuk spektrogram log-mel 2D (64x98) multi-label."""

    def __init__(self, n_keluaran: int = len(data.LABEL), dropout: float = 0.3):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.backbone.classifier[0].in_features  # 576
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_keluaran),
        )
        self.ambang = np.full(n_keluaran, 0.0, dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Jika masukan berupa spektrogram 1 kanal (N, 64, 98), ekspansi ke 3 kanal
        if x.ndim == 3:
            x = x[:, None, :, :].expand(-1, 3, -1, -1)
        elif x.ndim == 4 and x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return self.backbone(x)


def jalankan(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray | None = None,
    seed: int = 1,
    latih=None,
    cetak: str = "123450076",
    keluaran: int = len(data.LABEL),
    kodek=None,
) -> tuple[nn.Module, dict]:
    """Latih model transfer learning dan kalibrasi ambang keputusan dari data validasi."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ModelTL(n_keluaran=keluaran, dropout=HIPER["dropout"])

    # Discriminative fine-tuning: laju pembelajaran berbeda untuk backbone dan head
    param_groups = [
        {"params": model.backbone.features.parameters(), "lr": HIPER["lr_backbone"]},
        {"params": model.backbone.classifier.parameters(), "lr": HIPER["lr_head"]},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=HIPER["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HIPER["epochs"])

    X_train_t = torch.tensor(Xtr, dtype=torch.float32)[:, None, :, :].expand(-1, 3, -1, -1)
    y_train_t = torch.tensor(ytr, dtype=torch.float32)
    X_val_t = torch.tensor(Xva, dtype=torch.float32)[:, None, :, :].expand(-1, 3, -1, -1)

    N = len(X_train_t)
    batch_size = HIPER["batch_size"]
    epochs = HIPER["epochs"]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            logits = model(batch_x)
            diketahui = batch_y != data.ANOTASI_TIDAK_PASTI
            if diketahui.any():
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[diketahui], batch_y[diketahui]
                )
                loss.backward()
                opt.step()
        scheduler.step()

    # Evaluasi dan penalaan ambang batas keputusan adaptif pada data validasi
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    best_th_logit = np.full(keluaran, 0.0, dtype=np.float32)
    for c in range(keluaran):
        dikenal = yva[:, c] != data.ANOTASI_TIDAK_PASTI
        if not dikenal.any():
            continue
        y_col = yva[dikenal, c]
        pos = int(y_col.sum())
        if pos == 0:
            continue
        probs_col = val_probs[dikenal, c]
        best_f1 = -1.0
        best_th_prob = 0.5
        for th in np.linspace(0.2, 0.8, 13):
            pred_col = (probs_col >= th).astype(int)
            penyebut = pos + pred_col.sum()
            f1 = 2 * np.sum((y_col == 1) & (pred_col == 1)) / penyebut if penyebut > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_th_prob = th

        # Konversi ambang probabilitas ke ruang logit: log(p / (1 - p))
        best_th_prob = np.clip(best_th_prob, 1e-4, 1.0 - 1e-4)
        best_th_logit[c] = float(np.log(best_th_prob / (1.0 - best_th_prob)))

    model.ambang = best_th_logit

    info = {
        "ambang_logit": [round(float(v), 4) for v in best_th_logit],
        "arsitektur": HIPER["arsitektur"],
    }
    return model, info
