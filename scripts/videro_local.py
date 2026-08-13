#!/usr/bin/env python3
"""Тот же videro.py, но транскрипт считает локальный whisper вместо SpeechCore.

Зачем обёртка, а не правка analyze.py: апстрим остаётся нетронутым, git pull
не конфликтует. analyze.analyze() зовёт transcribe() как глобал модуля, поэтому
подмена атрибута до вызова работает.

Смысл именно в полном прогоне: транскрипт нужен не только ради субтитров — речь
окна уходит в промпт vision-модели как контекст сцены (analyze.py, describe_scene),
так что с ним captions и topic получаются точнее, чем по одним кадрам.

Флаги — как у videro.py:
  python scripts/videro_local.py видео.mp4 --lang en
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze          # noqa: E402
import hub_retry        # noqa: E402
import redact           # noqa: E402

# parakeet по умолчанию: на лекции он вышел в 5.3 раза быстрее и не терял куски
# речи, которые whisper схлопывал. whisper остаётся ради языков вне тех 25,
# что знает parakeet — ASR_BACKEND=whisper
BACKEND = os.getenv("ASR_BACKEND", "parakeet").lower()
if BACKEND == "whisper":
    import asr_local as asr      # noqa: E402
else:
    import asr_parakeet as asr   # noqa: E402

analyze.transcribe = asr.transcribe
hub_retry.apply(analyze)


def _analyze_and_redact(*args, **kwargs):
    """Вычистить секреты ДО того, как videro.py запишет timeline.json на диск.

    Нужно потому, что утечку создаёт сам пайплайн: fix_subtitles «исправляет»
    реплики по OCR-контексту и переносит в транскрипт токены с экрана. Ручной
    прогон redact.py забывается, а файл к тому моменту уже лежит.
    Отключается через NO_REDACT=1.
    """
    tl = _analyze_orig(*args, **kwargs)
    if os.getenv("NO_REDACT"):
        print("  ! NO_REDACT=1 — очистка секретов пропущена", flush=True)
        return tl
    found = redact.redact_timeline(tl)
    if found:
        print(f"\n  ! вычищено секретов: {len(found)} (значения заменены маркером, "
              f"имена переменных сохранены)")
        for start, where, hits in found[:12]:
            print(f"      {redact.fmt_ts(start):>9}  {where:30s} {', '.join(sorted(set(hits)))}")
        if len(found) > 12:
            print(f"      … ещё {len(found) - 12}")
    return tl


_analyze_orig = analyze.analyze
analyze.analyze = _analyze_and_redact

import videro           # noqa: E402

if __name__ == "__main__":
    print(f"ASR: {BACKEND} ({asr.MODEL}), диаризации нет — её кладёт diarize.py")
    videro.main()
