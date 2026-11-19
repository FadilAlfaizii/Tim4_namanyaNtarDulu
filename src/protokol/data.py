"""Pemuat audio multi-label Tim 04 dari arsip sesi meja."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import wave
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import resample_poly

from src.common import manifest, split

TUGAS = "klasifikasi_multilabel"
AKSI = ("buka_pintu", "tutup_pintu", "ketuk_meja", "letakkan_gelas")
ALAT = ("kipas", "pompa", "buzzer")
LABEL = AKSI + ALAT
ANOTASI_TIDAK_PASTI = -1
RATE_ASLI = 48000
RATE = 16000
DETIK = (1, 2, 3)
VERSI = 1
# Gangguan perekam yang beririsan dengan kondisi di log sesi.
KONDISI_TERGANGGU = (2, 3, 4, 92, 103, 104, 176, 210)


@dataclass(frozen=True)
class Jendela:
    kondisi: int
    ulangan: int
    detik: int
    aksi_isyarat: str
    mulai_host: float
    berkas: str
    sampel_awal: int
    audio: np.ndarray
    target_alat: tuple[int, ...]

    @property
    def id(self) -> str:
        return f"{self.kondisi}:{self.detik}"


def grup_split() -> tuple[tuple[int, ...], ...]:
    ulangan = np.arange(1, 7)
    sisa, uji = split.indeks(len(ulangan), porsi_val=1 / 3)
    # Urutan sisa berasal dari pengacakan beku. Dua grup pertama untuk validasi.
    return tuple(tuple(map(int, ulangan[i])) for i in (sisa[2:], sisa[:2], uji))


def data_hash() -> str:
    parameter = {
        "versi": VERSI, "seed_split": split.SEED_SPLIT, "grup": grup_split(),
        "label": LABEL, "detik": DETIK, "kondisi_terganggu": KONDISI_TERGANGGU,
    }
    return manifest.data_hash(json.dumps(parameter, sort_keys=True).encode("utf-8"))


@lru_cache(maxsize=1)
def ekstrak() -> tuple[Jendela, ...]:
    m = manifest.baca()
    arsip = manifest.AKAR / m["arsip"]
    if not arsip.is_file():
        raise FileNotFoundError(f"Arsip sesi belum tersedia di {arsip}. Lihat data/MANIFEST.json.")
    with arsip.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != m["sha256_arsip"]:
            raise ValueError(f"SHA-256 arsip berbeda dari manifest: {arsip}")
        f.seek(0)
        with tarfile.open(fileobj=f, mode="r:gz") as paket:
            return _baca_sesi(paket, m["sesi"])


def _baca_sesi(paket: tarfile.TarFile, sesi: str) -> tuple[Jendela, ...]:
    def baca_csv(nama):
        with paket.extractfile(f"{sesi}/{nama}") as f:
            return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")))

    kondisi = baca_csv("label.csv")
    if len(kondisi) != 240 or {int(r["no"]) for r in kondisi} != set(range(1, 241)):
        raise ValueError("Identitas kondisi sesi berbeda dari protokol Tim 04.")
    rekaman = []
    indeks = sorted(baca_csv("audio_index.csv"), key=lambda r: float(r["t_mulai_host"]))
    for r in indeks:
        nama = r["berkas"]
        with paket.extractfile(f"{sesi}/audio/{nama}") as f, wave.open(f) as wav:
            format_wav = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
            n = int(r["jumlah_sampel"])
            if format_wav != (RATE_ASLI, 1, 2) or wav.getnframes() != n or int(r["sample_rate"]) != RATE_ASLI:
                raise ValueError(f"Format atau panjang WAV tidak cocok dengan indeks: {nama}")
            pcm = np.frombuffer(wav.readframes(n), dtype="<i2")
            if len(pcm) != n:
                raise ValueError(f"WAV terpotong: {nama}")
            rekaman.append(pcm)
    mulai_wav = [float(r["t_mulai_host"]) for r in indeks]
    aktuator = defaultdict(list)
    for r in baca_csv("aktuator.csv"):
        aktuator[r["nama"]].append((float(r["t_host"]), int(r["nilai"])))
    for perintah in aktuator.values():
        perintah.sort()

    hasil = []
    for r in kondisi:
        nomor = int(r["no"])
        if nomor in KONDISI_TERGANGGU:
            continue
        mulai = float(r["t_mulai_host"])
        if not _alat_cocok(aktuator, r, mulai):
            continue
        alat = tuple(int(int(r[k]) > 0) for k in ALAT)
        akhir = float(r["t_selesai_host"])
        for detik in DETIK:
            waktu = mulai + detik
            i = bisect_right(mulai_wav, waktu) - 1
            if i < 0 or waktu + 1 > akhir:
                continue
            awal = round((waktu - mulai_wav[i]) * RATE_ASLI)
            akhir_sampel = awal + RATE_ASLI
            if akhir_sampel > len(rekaman[i]):
                continue
            pcm = rekaman[i][awal:akhir_sampel].astype(np.float32) / 32768
            audio = resample_poly(pcm, RATE, RATE_ASLI)
            audio.setflags(write=False)
            hasil.append(Jendela(nomor, int(r["ulangan"]), detik, r["aksi"], waktu,
                                 indeks[i]["berkas"], awal, audio, alat))
    return tuple(hasil)


def _alat_cocok(aktuator, kondisi, mulai):
    """Cari konfirmasi setelan alat di sekitar awal kondisi."""
    for nama in ALAT:
        catatan = aktuator[nama]
        batas = bisect_left(catatan, (mulai + 1,))
        if not batas:
            return False
        waktu, nilai = catatan[batas - 1]
        if waktu < mulai - 0.5 or nilai != int(kondisi[nama]):
            return False
    return True


def _log_mel(audio: np.ndarray) -> np.ndarray:
    frame = np.lib.stride_tricks.sliding_window_view(audio, 400, axis=-1)[:, ::160]
    spektrum = np.fft.rfft(frame * np.hanning(400), n=512, axis=-1)
    daya = np.abs(spektrum) ** 2
    hz = np.fft.rfftfreq(512, 1 / RATE)
    mel = 2595 * np.log10(1 + hz / 700)
    pusat = np.linspace(mel[0], mel[-1], 66)
    kiri = (mel - pusat[:-2, None]) / (pusat[1:-1, None] - pusat[:-2, None])
    kanan = (pusat[2:, None] - mel) / (pusat[2:, None] - pusat[1:-1, None])
    bank = np.maximum(0, np.minimum(kiri, kanan))
    return np.log(np.maximum(daya @ bank.T, 1e-10)).transpose(0, 2, 1).astype(np.float32)


def _target(jendela: tuple[Jendela, ...]) -> tuple[np.ndarray, np.ndarray]:
    lokasi = manifest.AKAR / manifest.baca()["anotasi_aksi"]
    anotasi = json.loads(lokasi.read_text(encoding="utf-8"))
    if not isinstance(anotasi, dict):
        raise ValueError("Anotasi aksi harus berupa objek JSON.")
    tersisa = anotasi.keys() - {j.id for j in jendela}
    if tersisa:
        raise ValueError(f"ID anotasi bukan jendela yang tersedia: {sorted(tersisa)}")
    target = []
    pakai = []
    for j in jendela:
        isi = anotasi.get(j.id)
        pakai.append(isi != "keluarkan")
        if isi is None or isi == "keluarkan":
            aksi = [ANOTASI_TIDAK_PASTI] * len(AKSI)
        elif isinstance(isi, list) and all(isinstance(k, str) and k in AKSI for k in isi):
            aksi = [int(k in isi) for k in AKSI]
        else:
            raise ValueError(f"Anotasi aksi tidak sah untuk jendela {j.id}: {isi!r}")
        target.append(aksi + list(j.target_alat))
    return np.asarray(target, dtype=np.int8), np.asarray(pakai)


def indeks_data(*, sertakan_uji: bool = True):
    jendela = ekstrak()
    target, pakai = _target(jendela)
    ulangan = np.asarray([j.ulangan for j in jendela])
    grup = grup_split()
    for nama, g in zip(("latih", "validasi", "uji"), grup if sertakan_uji else grup[:2]):
        masker = pakai & np.isin(ulangan, g)
        if not masker.any():
            raise ValueError(f"Split {nama} tidak berisi jendela yang dapat dipakai.")
        yield jendela, target, masker


def muat(*, normalisasi: bool = True, sertakan_uji: bool = True):
    """Kembalikan X dan y untuk latih, validasi, uji, berurutan.

    X berbentuk (N, 64, 98). y berbentuk (N, 7) dengan nilai 0, 1, atau -1
    untuk label aksi yang belum diketahui. Saat sertakan_uji=False, dua nilai
    terakhir adalah None. Baseline mengambil log-mel sebelum normalisasi.
    """
    hasil = []
    for jendela, target, masker in indeks_data(sertakan_uji=sertakan_uji):
        audio = np.stack([j.audio for j, pakai in zip(jendela, masker) if pakai])
        hasil.extend((_log_mel(audio), target[masker]))
    if normalisasi:
        rata = hasil[0].mean(axis=(0, 2), dtype=np.float64, keepdims=True)
        simpangan = hasil[0].std(axis=(0, 2), dtype=np.float64, keepdims=True).clip(min=1e-6)
        for i in range(0, len(hasil), 2):
            hasil[i] = ((hasil[i] - rata) / simpangan).astype(np.float32)
    if not sertakan_uji:
        hasil.extend((None, None))
    return tuple(hasil)
