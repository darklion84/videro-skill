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
import scene_cache      # noqa: E402

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

# Потолок вывода для vision. Апстрим ставит 1600, а плотный экран с кодом требует
# ~1520 — впритык. Ответ, не влезший в лимит, приходит ПУСТЫМ (finish=length,
# content отсутствует), json.loads падает, и сцена превращается в заглушку.
# На Lecture 2 так осыпалось 128 сцен из 273; на первой лекции с простыми
# экранами — только 8. Поднимаем запас, а не гадаем.
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "4000"))
_session_post = analyze.requests.post


def _post_with_headroom(url, **kw):
    body = kw.get("json")
    if isinstance(body, dict) and body.get("max_tokens", 0) < VISION_MAX_TOKENS:
        body["max_tokens"] = VISION_MAX_TOKENS
    return _session_post(url, **kw)


analyze.requests.post = _post_with_headroom


_describe_orig = analyze.describe_scene


def _is_junk(res: dict) -> bool:
    """Ответ, который нельзя ни показывать, ни кэшировать.

    Под нагрузкой модель периодически отдаёт пустоту: content приходит пустой,
    json.loads падает, и апстрим подставляет заглушку со всеми пустыми полями.
    На Lecture 2 так осыпалось 128 сцен из 273 — почти половина, и по ним потом
    нечего было резать на главы. Кадры при этом нормальные: повторный вызов с
    теми же параметрами описывает сцену без проблем.
    """
    return (str(res.get("action", "")).startswith("[error:")
            or not (res.get("caption") or "").strip())


def _describe_cached(frames, speech, start, end, tries=3):
    """Дорогой vision-вызов через кэш на диске, с повтором на пустой ответ.

    Результат ложится в файл сразу, поэтому убитый процесс теряет одну сцену,
    а не весь прогон. Мусорные ответы не кэшируем — иначе разовый сбой
    заморозится в файле и будет отдаваться при каждом повторе.
    """
    k = scene_cache.key(start, end, analyze.VISION_MODEL)
    hit = scene_cache.get(k)
    if hit is not None:
        return hit
    res = {}
    for _ in range(tries):
        res = _describe_orig(frames, speech, start, end)
        if not _is_junk(res):
            scene_cache.put(k, res)
            return res
    return res


analyze.describe_scene = _describe_cached


def _no_fix_subtitles(transcript, scenes, batch=20):
    """Апстримный шаг коррекции субтитров выключен. Три причины:

    1. Именно он занёс в транскрипт JIRA_TOKEN с экрана — «исправлял» реплику
       по OCR-контексту и подставил туда настоящий секрет.
    2. Его работу делает normalize.py: явный глоссарий вместо догадок по экрану
       плюс детерминированный фильтр, отклоняющий правки не по словарю.
    3. На двухчасовой лекции процесс дважды умирал ровно на этом шаге.

    Вернуть: FIX_SUBTITLES=1.
    """
    print("  коррекция субтитров пропущена (её делает normalize.py)", flush=True)


if not os.getenv("FIX_SUBTITLES"):
    analyze.fix_subtitles = _no_fix_subtitles


def _analyze_and_redact(*args, **kwargs):
    """Вычистить секреты ДО того, как videro.py запишет timeline.json на диск.

    Нужно потому, что утечку создаёт сам пайплайн: fix_subtitles «исправляет»
    реплики по OCR-контексту и переносит в транскрипт токены с экрана. Ручной
    прогон redact.py забывается, а файл к тому моменту уже лежит.
    Отключается через NO_REDACT=1.
    """
    video_path = args[0] if args else kwargs.get("video_path", "")
    have = scene_cache.load(scene_cache.path_for(video_path))
    if have:
        print(f"  кэш сцен: {have} готовых, за них платить не будем", flush=True)

    tl = _analyze_orig(*args, **kwargs)

    from_cache, computed = scene_cache.stats()
    if from_cache:
        print(f"  сцен взято из кэша: {from_cache}, посчитано заново: {computed}")
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
