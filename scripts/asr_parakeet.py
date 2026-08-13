#!/usr/bin/env python3
"""ASR на parakeet-tdt-0.6b-v3 (NVIDIA, порт под MLX). Бэкенд по умолчанию.

Замерено на двухчасовой лекции против whisper large-v3 на той же дорожке:
  скорость    204 с (36.8x) против 1078 с (6.9x)   — в 5.3 раза быстрее
  полнота     whisper потерял ~308 слов, которые parakeet распознал; худший
              случай — 67 слов связной речи схлопнулись в «6 раз»
  пунктуация  132 точки на 1000 слов против 49
  термины     после нормализации калек осталось 7 против 16

Ограничение, ради которого whisper остаётся запасным: parakeet знает 25
европейских языков, whisper около 99. Переключение: ASR_BACKEND=whisper.

Язык модель определяет сама, параметр lang принимается только ради общего
интерфейса с asr_local.transcribe и не используется.

Env: PARAKEET_MODEL, PARAKEET_CHUNK (сек), PARAKEET_OVERLAP (сек).
"""
from __future__ import annotations

import os

MODEL = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
CHUNK = float(os.getenv("PARAKEET_CHUNK", "120"))
OVERLAP = float(os.getenv("PARAKEET_OVERLAP", "15"))


def transcribe(audio_path: str, lang: str = "", glossary: str = "") -> list[dict]:
    """Аудио → сегменты [{start, end, text, speaker}]. Сигнатура как у whisper-версии.

    glossary игнорируется: в parakeet-mlx нет API подсказок (ни prompt, ни boost,
    ни hotwords — проверено по исходникам пакета). Термины чинит normalize.py.
    """
    from parakeet_mlx import from_pretrained

    model = from_pretrained(MODEL)
    res = model.transcribe(audio_path, chunk_duration=CHUNK, overlap_duration=OVERLAP)
    return [{"start": round(s.start, 2), "end": round(s.end, 2),
             "text": s.text.strip(), "speaker": None}
            for s in getattr(res, "sentences", []) if s.text.strip()]


if __name__ == "__main__":
    import argparse
    import json
    import time

    ap = argparse.ArgumentParser(description="parakeet → сегменты с таймкодами")
    ap.add_argument("media", help="аудио или видео (что читает ffmpeg)")
    ap.add_argument("--json", default=None, help="куда сложить сегменты")
    args = ap.parse_args()

    print(f"▶ {MODEL}\n  {args.media}")
    t0 = time.time()
    segs = transcribe(args.media)
    dt = time.time() - t0
    span = segs[-1]["end"] if segs else 0
    print(f"\n✔ {len(segs)} сегментов за {dt:.0f}s "
          f"(речь до {span:.0f}s → {span / dt if dt else 0:.1f}× реалтайма)")
    for s in segs[:5]:
        print(f"  [{s['start']:7.2f}–{s['end']:7.2f}] {s['text'][:80]}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(segs, f, ensure_ascii=False, indent=2)
        print(f"  → {args.json}")
