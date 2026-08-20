#!/usr/bin/env python3
"""Кэш разметки сцен: дорогие vision-вызовы переживают обрыв процесса.

Зачем: analyze.analyze() держит результат в памяти и отдаёт его целиком, а
videro.py пишет timeline.json только в самом конце. На двухчасовой лекции это
273 vision-вызова и полтора часа, и если процесс убили на предпоследнем шаге —
теряется всё. Так и случилось с Lecture 2.

Формат — JSONL с дозаписью: каждая готовая сцена ложится отдельной строкой сразу.
Обрыв в любой момент теряет максимум одну сцену, а не весь прогон. Повторный
запуск читает файл и не платит за то, что уже посчитано.

Ключ — границы сцены плюс модель: сменишь модель, и кэш не подсунет чужое.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

_lock = threading.Lock()          # analyze зовёт describe_scene из пула потоков
_path: str | None = None
_hits: dict[str, dict] = {}
_new = 0


def key(start: float, end: float, model: str) -> str:
    return f"{start:.2f}-{end:.2f}-{model}"


def path_for(video: str, out_dir: str = "") -> str:
    """Кэш живёт рядом с прогоном, а если папка неизвестна — в общем месте."""
    if out_dir:
        return os.path.join(out_dir, ".scenes.jsonl")
    h = hashlib.sha1(os.path.abspath(video).encode()).hexdigest()[:12]
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videro-out")
    os.makedirs(os.path.join(root, ".cache"), exist_ok=True)
    return os.path.join(root, ".cache", f"scenes-{h}.jsonl")


def load(p: str) -> int:
    global _path, _hits, _new
    _path, _hits, _new = p, {}, 0
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                _hits[rec["k"]] = rec["v"]
            except (json.JSONDecodeError, KeyError):
                continue          # оборванная последняя строка — не беда
    return len(_hits)


def get(k: str) -> dict | None:
    return _hits.get(k)


def put(k: str, value: dict) -> None:
    global _new
    if _path is None:
        return
    with _lock:
        _hits[k] = value
        _new += 1
        with open(_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": value}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())     # ради переживания kill, а не только исключения


def stats() -> tuple[int, int]:
    """→ (взято из кэша, посчитано заново)"""
    return len(_hits) - _new, _new
