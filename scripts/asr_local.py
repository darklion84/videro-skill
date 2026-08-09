#!/usr/bin/env python3
"""Локальный ASR на mlx-whisper — замена SpeechCore, пока тот недоступен.

Отдаёт транскрипт в том же формате, что и analyze.transcribe(), поэтому
подставляется вместо неё без правок остального пайплайна (см. videro_local.py).

Отличие от SpeechCore: диаризации нет, поле speaker всегда None. Whisper
размечает речь, но не говорящих; для этого нужен отдельный pyannote.

Env: WHISPER_MODEL (репо весов), WHISPER_MLX_VERBOSE=1 (прогресс от mlx).

Отдельно, чтобы проверить распознавание без трат на vision:
  python scripts/asr_local.py видео.mp4 --lang en
"""
from __future__ import annotations

import os

MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx")


def transcribe(audio_path: str, lang: str) -> list[dict]:
    """Аудио → сегменты [{start, end, text, speaker}]. Сигнатура как у SpeechCore-версии."""
    import mlx_whisper

    res = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=MODEL,
        language=lang or None,          # пусто → whisper определит язык сам
        verbose=True if os.getenv("WHISPER_MLX_VERBOSE") else None,
        condition_on_previous_text=False,   # иначе whisper зацикливается на длинных паузах
    )
    out = []
    for seg in res.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({"start": float(seg.get("start", 0) or 0),
                    "end": float(seg.get("end", 0) or 0),
                    "text": text,
                    "speaker": None})
    return out


if __name__ == "__main__":
    import argparse
    import json
    import time

    ap = argparse.ArgumentParser(description="локальный whisper → сегменты с таймкодами")
    ap.add_argument("media", help="аудио или видео (что читает ffmpeg)")
    ap.add_argument("--lang", default="", help="язык речи; пусто — автоопределение")
    ap.add_argument("--json", default=None, help="куда сложить сегменты (по умолчанию только сводка)")
    args = ap.parse_args()

    print(f"▶ {MODEL}\n  {args.media}")
    t0 = time.time()
    segs = transcribe(args.media, args.lang)
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
