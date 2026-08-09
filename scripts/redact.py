#!/usr/bin/env python3
"""Вычистить секреты из готовой разметки: токены, ключи, почты.

Зачем: секрет попадает в текст не только когда его произносят вслух. На лекции
whisper услышал описание словами («джей подчёркивание токен равно токен который
вы вставили»), а апстримный fix_subtitles «исправил» реплику по OCR-контексту и
подставил туда настоящий токен с экрана. То есть утечку делает сам пайплайн.

Имена переменных остаются — они и есть содержание урока. Заменяется значение.

По умолчанию чистит transcript, text_raw и субтитры. OCR-поле сцен
(on_screen_text) — флагом --ocr: оно уходит в модель при каждом прогоне
chapters.py, так что секрет в нём продолжает утекать наружу.

  python scripts/redact.py videro-out/x/timeline.json --dry-run
  python scripts/redact.py videro-out/x/timeline.json --ocr
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

MARK = "⚠ВОЗМОЖНА УТЕЧКА, ЗАМЕНИТЕ КЛЮЧ⚠"

# Сначала — узнаваемые по форме секреты, они однозначны и ловятся без контекста.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Atlassian",   re.compile(r"\bATATT[A-Za-z0-9_\-=.]{16,}")),
    ("JWT",         re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("OpenAI-like", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("GitHub",      re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}")),
    ("AWS",         re.compile(r"\bAKIA[0-9A-Z]{12,}")),
    ("Google",      re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")),
    ("Slack",       re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("Bearer",      re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=]{16,}")),
    ("почта",       re.compile(r"\b[\w.+\-]+@[\w\-]+\.[A-Za-z]{2,}\b")),
]

# Затем — ЛЮБОЕ значение у переменной вида KEY=... / KEY: ...
# Имя переменной сохраняется, значение вырезается.
KV = re.compile(r"\b([A-Z][A-Z0-9_]{2,})(\s*[=:]\s*)([^\s,;]{8,})")

# И наконец — улов на секреты неизвестного формата. Код авторизации Claude Code
# (92 символа без префикса и без «КЛЮЧ=») не ловился ничем из перечисленного.
# Требуем длину, оба регистра и цифру: это отсекает git-хеши (только нижний
# регистр), пути (в них есть слэш) и обычные длинные слова.
BLOB = re.compile(r"\b(?=[A-Za-z0-9_\-+=]{32,}\b)(?=[^\s]*[a-z])(?=[^\s]*[A-Z])"
                  r"(?=[^\s]*\d)[A-Za-z0-9_\-+=]{32,}\b")


def redact_text(s: str) -> tuple[str, list[str]]:
    """→ (очищенный текст, какие виды секретов нашлись)"""
    if not s:
        return s, []
    hits = []
    for name, rx in PATTERNS:
        s, n = rx.subn(MARK, s)
        if n:
            hits += [name] * n

    def kv(m):
        # ищем знак маркера, а не маркер целиком: значение в KV ловится до первого
        # пробела или запятой, а в маркере есть и то и другое — полное совпадение
        # не срабатывало и текст заменялся повторно
        if "⚠" in m.group(3):
            return m.group(0)
        hits.append(f"{m.group(1)}=…")
        return f"{m.group(1)}{m.group(2)}{MARK}"

    s = KV.sub(kv, s)

    s, n = BLOB.subn(MARK, s)
    if n:
        hits += ["длинная строка"] * n
    return s, hits


def redact_timeline(tl: dict, ocr: bool = True) -> list[tuple[float, str, list[str]]]:
    """Почистить таймлайн на месте. → [(таймкод, где, что нашлось)].

    Вызывается и из CLI, и автоматически в конце прогона (videro_local.py):
    ручной шаг забывается, а секрет попадает в файл сам собой.
    """
    found = []
    for seg in tl.get("transcript") or []:
        for field in ("text", "text_raw"):
            if field in seg:
                seg[field], hits = redact_text(seg[field])
                if hits:
                    found.append((seg["start"], f"транскрипт.{field}", hits))
    if ocr:
        for i, sc in enumerate(tl.get("scenes") or []):
            if sc.get("on_screen_text"):
                sc["on_screen_text"], hits = redact_text(sc["on_screen_text"])
                if hits:
                    found.append((sc["start"], f"сцена #{i}.on_screen_text", hits))
    if tl.get("srt"):
        tl["srt"], hits = redact_text(tl["srt"])
        if hits:
            found.append((0.0, "srt", hits))
    return found


def fmt_ts(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser(description="вычистить секреты из timeline.json и субтитров")
    ap.add_argument("timeline")
    ap.add_argument("--ocr", action="store_true",
                    help="чистить и on_screen_text сцен (он уходит в модель при каждом прогоне)")
    ap.add_argument("--dry-run", action="store_true", help="только показать находки, ничего не писать")
    args = ap.parse_args()

    path = os.path.abspath(args.timeline)
    out_dir = os.path.dirname(path)
    with open(path, encoding="utf-8") as f:
        tl = json.load(f)

    # dry-run работает на копии: сама функция чистит на месте
    target = json.loads(json.dumps(tl)) if args.dry_run else tl
    found = redact_timeline(target, ocr=args.ocr)

    srt_path = os.path.join(out_dir, "subtitles.srt")
    srt_hits: list[str] = []
    if not tl.get("srt") and os.path.exists(srt_path):    # srt лежит только файлом
        with open(srt_path, encoding="utf-8") as f:
            new_srt, srt_hits = redact_text(f.read())
        if srt_hits and not args.dry_run:
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(new_srt)

    if not found and not srt_hits:
        print("✔ секретов не найдено")
        return

    print(f"{'НАЙДЕНО (ничего не записано)' if args.dry_run else 'ВЫЧИЩЕНО'}:")
    for start, where, hits in found:
        print(f"  {fmt_ts(start):>9}  {where:32s} {', '.join(sorted(set(hits)))}")
    if srt_hits:
        print(f"  {'':>9}  subtitles.srt{'':20s} {', '.join(sorted(set(srt_hits)))}")

    if args.dry_run:
        print("\nчтобы применить — повтори без --dry-run")
        return

    if tl.get("srt"):                                     # переписать файл субтитров
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(tl["srt"])

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tl, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"\n✔ {path}")
    if not args.ocr:
        print("  ВНИМАНИЕ: on_screen_text сцен не тронут — секрет там остался и будет "
              "уходить в модель при каждом прогоне chapters.py. Повтори с --ocr.", file=sys.stderr)


if __name__ == "__main__":
    main()
