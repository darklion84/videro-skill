#!/usr/bin/env python3
"""Локальный сервер плеера: статика с поддержкой Range + запись имён спикеров.

Чем не устраивает `python -m http.server`:
  1. нет Range-запросов, а без них браузер не перематывает большие mp4 —
     на двухчасовой записи клик по главе просто не работает;
  2. только чтение, а кнопке «Назвать спикеров» нужно куда-то сохранять.

API (только с 127.0.0.1, только внутри каталога раздачи):
  GET  /api/speakers?src=/videro-out/x/   → реестр спикеров с примерами реплик
  POST /api/speakers  {src, names}        → записать имена и разнести их
                                            по transcript и subtitles.srt

Путь из запроса приводится к абсолютному и проверяется на выход за корень раздачи,
иначе POST на сервер, слушающий localhost, стал бы записью в произвольный файл.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speakers        # noqa: E402

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".m3u8": "application/vnd.apple.mpegurl",
                      ".ts": "video/mp2t",
                      ".vtt": "text/vtt"}

    # ── API ──────────────────────────────────────────────────────────────────
    def _root(self) -> Path:
        return Path(self.directory).resolve()

    @staticmethod
    def _utf8(s: str) -> str:
        """http.server разбирает строку запроса как latin-1 — кириллица в пути бьётся."""
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s

    def _timeline(self, src: str) -> Path:
        """src из запроса → путь к timeline.json, не выпуская за корень раздачи."""
        if not src:
            raise ValueError("не передан src")
        p = (self._root() / src.lstrip("/")).resolve()
        if not p.is_relative_to(self._root()):
            raise ValueError("src ведёт за пределы каталога раздачи")
        tl = p / "timeline.json"
        if not tl.is_file():
            raise ValueError(f"нет timeline.json в {src}")
        return tl

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/api/speakers":
            src = self._utf8((parse_qs(urlparse(self.path).query).get("src") or [""])[0])
            try:
                return self._json(speakers.make_roster(str(self._timeline(src))))
            except (ValueError, OSError) as e:
                return self._json({"error": str(e)}, 400)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/speakers":
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1 << 20:
                raise ValueError("слишком большое тело запроса")
            req = json.loads(self.rfile.read(n) or b"{}")
            tl = str(self._timeline(req.get("src") or ""))
            names = req.get("names") or {}
            if not isinstance(names, dict):
                raise ValueError("names должен быть объектом {SPEAKER_XX: имя}")
            speakers.save_names(tl, {str(k): str(v) for k, v in names.items()})
            res = speakers.apply_names(tl, srt_names=bool(req.get("srt_names", True)))
            res["roster"] = speakers.make_roster(tl)
            return self._json(res)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            return self._json({"error": str(e)}, 400)

    # ── Range: без него не перематывается большое видео ──────────────────────
    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.isfile(path):
            return super().send_head()
        m = RANGE_RE.fullmatch(rng.strip())
        if not m:
            return super().send_head()

        size = os.path.getsize(path)
        first, last = m.group(1), m.group(2)
        if first == "":                                   # bytes=-N — последние N байт
            length = min(int(last or 0), size)
            start, end = size - length, size - 1
        else:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        if start >= size or start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        f = open(path, "rb")
        f.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()                     # Accept-Ranges добавляется там
        return _Slice(f, end - start + 1)

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Accept-Ranges", "bytes")
            # timeline.json и speakers.json меняются прямо во время работы страницы:
            # закэшированная копия показала бы старые имена после сохранения
            if urlparse(self.path).path.endswith(".json"):
                self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *a):                        # без шума от сотен сегментов видео
        if "/api/" in self.path or int(str(a[1] if len(a) > 1 else 200)[0]) >= 4:
            super().log_message(fmt, *a)


class _Slice:
    """Обёртка над файлом, отдающая ровно N байт (copyfile читает до EOF)."""

    def __init__(self, f, length):
        self.f, self.left = f, length

    def read(self, n=-1):
        if self.left <= 0:
            return b""
        chunk = self.f.read(self.left if n < 0 else min(n, self.left))
        self.left -= len(chunk)
        return chunk

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser(description="сервер плеера videro")
    ap.add_argument("root", help="каталог раздачи (корень репозитория)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    handler = partial(Handler, directory=os.path.abspath(args.root))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлен")


if __name__ == "__main__":
    main()
