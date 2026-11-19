"""Read-only audit for a Tim 04 session folder. Standard library only.

Usage: python fieldwork/audit_session.py --session PATH [--expected 240]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import wave
from pathlib import Path


def csv_info(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        count = sum(1 for row in reader if any(cell.strip() for cell in row))
    return header, count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=240)
    args = parser.parse_args()
    folder = args.session.resolve()
    if not folder.is_dir():
        parser.error(f"Folder sesi tidak ada: {folder}")

    print(f"Folder sesi: {folder}")
    problems = []
    for name in ("manifest.json", "label.csv", "audio_index.csv", "digital.csv",
                 "imu.csv", "daya.csv", "lingkungan.csv", "aktuator.csv"):
        path = folder / name
        if not path.is_file():
            print(f"TIDAK ADA: {name}")
            if name in ("manifest.json", "label.csv", "audio_index.csv"):
                problems.append(name)
            continue
        if path.suffix == ".csv":
            try:
                header, count = csv_info(path)
                print(f"{name}: {count} baris data; kolom: {', '.join(header)}")
                if name == "label.csv" and count != args.expected:
                    problems.append(f"label.csv: {count}, target {args.expected}")
            except (OSError, UnicodeError, csv.Error) as exc:
                print(f"GAGAL BACA {name}: {exc}")
                problems.append(name)
        else:
            print(f"ADA: {name} ({path.stat().st_size} byte)")

    label = folder / "label.csv"
    if label.is_file():
        print(f"SHA-256 label.csv: {sha256(label)}")

    wavs = sorted(folder.rglob("*.wav"))
    print(f"Jumlah WAV: {len(wavs)}")
    if not wavs:
        problems.append("tidak ada WAV")
    rates = {}
    total_seconds = 0.0
    for path in wavs:
        try:
            with wave.open(str(path), "rb") as audio:
                rate = audio.getframerate()
                channels = audio.getnchannels()
                seconds = audio.getnframes() / rate if rate else 0
                rates[(rate, channels)] = rates.get((rate, channels), 0) + 1
                total_seconds += seconds
                if rate != 16000 or channels != 1:
                    problems.append(f"format WAV: {path.relative_to(folder)} ({rate} Hz, {channels} kanal)")
        except (OSError, EOFError, wave.Error) as exc:
            problems.append(f"WAV rusak/tidak didukung: {path.relative_to(folder)} ({exc})")
    print(f"Format WAV (Hz, kanal -> jumlah): {rates}")
    print(f"Total durasi berkas WAV: {total_seconds:.1f} detik (belum memeriksa tumpang tindih)")
    print("Status:", "PERLU DITINJAU: " + "; ".join(problems) if problems else "audit dasar lulus")
    print("Lanjutkan dengar sampel, cek sinkronisasi t_host, sensor per kondisi, dan label aksi secara manual.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
