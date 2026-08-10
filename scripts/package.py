#!/usr/bin/env python3
"""Собрать готовый прогон в архив, который коллега запускает двойным кликом.

Что кладётся: видео, разметка, субтитры, плеер и маленький сервер на стандартной
библиотеке. Ни venv, ни ключей, ни интернета получателю не нужно — только Python 3,
который на macOS и Linux есть из коробки.

Почему нельзя просто открыть index.html из файла: браузер запрещает fetch с file://,
поэтому timeline.json не прочитается, и без Range-запросов не работает перемотка
большого mp4. Отсюда локальный сервер в комплекте.

  python scripts/package.py ai-in-ba-lecture1
  python scripts/package.py ai-in-ba-lecture1 --out ~/Desktop --no-zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hlsmake import dir_size, make_hls  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".jpg", ".jpeg", ".png", ".ts"}

LAUNCH_SH = """#!/bin/sh
# Запускает локальный плеер и открывает браузер.
cd "$(dirname "$0")"
command -v python3 >/dev/null 2>&1 || {{ echo "Нужен Python 3: https://www.python.org/downloads/"; read _; exit 1; }}
exec python3 serve.py . --port 0 --open "{url}"
"""

LAUNCH_BAT = """@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Nuzhen Python 3: https://www.python.org/downloads/ && pause && exit /b 1)
python serve.py . --port 0 --open "{url}"
pause
"""

READ_ME = """{title}

КАК ОТКРЫТЬ
-----------
macOS   — двойной клик по start-mac.command
          Если система ругается «неизвестный разработчик»: правый клик по файлу →
          «Открыть» → «Открыть» ещё раз. Это нужно только в первый раз.
Windows — двойной клик по start-windows.bat
Linux   — ./start-linux.sh

Откроется вкладка браузера с видео, главами, хайлайтами и транскриптом.
Чтобы закрыть — закрой окно терминала, которое открылось вместе с плеером.

ЧТО ВНУТРИ
----------
{stats}

Нужен установленный Python 3. На macOS и Linux он уже есть.
Интернет не нужен, всё работает локально.

КАК ПОЛЬЗОВАТЬСЯ
----------------
Клик по главе, хайлайту или реплике транскрипта — перемотка на это место.
Цветная полоса под видео — сцены; жёлтые отметки это хайлайты, по ней тоже можно кликать.
Тумблер «Синхронизация с видео» листает транскрипт за воспроизведением.
Субтитры включаются в меню плеера.
"""


def human(n: int) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description="запаковать прогон для раздачи")
    ap.add_argument("name", help="имя папки в videro-out")
    ap.add_argument("--out", default=None, help="куда положить (по умолчанию рядом, в videro-out)")
    ap.add_argument("--no-zip", action="store_true", help="оставить папкой, не жать в zip")
    ap.add_argument("--hls", action="store_true",
                    help="пережать видео в HLS под скринкаст: примерно вдвое легче, "
                         "и сегментами по паре МБ вместо одного большого файла")
    ap.add_argument("--bitrate", default="350k",
                    help="видеобитрейт 720p при --hls (default 350k)")
    args = ap.parse_args()

    src_dir = os.path.join(ROOT, "videro-out", args.name)
    tl_path = os.path.join(src_dir, "timeline.json")
    if not os.path.isfile(tl_path):
        sys.exit(f"нет {tl_path}")

    with open(tl_path, encoding="utf-8") as f:
        tl = json.load(f)
    meta = {}
    meta_path = os.path.join(src_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    out_base = os.path.abspath(args.out) if args.out else os.path.join(ROOT, "videro-out")
    pkg = os.path.join(out_base, f"{args.name}-share")
    shutil.rmtree(pkg, ignore_errors=True)
    data = os.path.join(pkg, "videro-out", args.name)
    os.makedirs(data, exist_ok=True)

    print(f"▶ собираю {pkg}")

    # разметка. Копируем по содержимому: video.mp4 у нас симлинк на исходник
    for fn in ("timeline.json", "subtitles.srt", "poster.jpg"):
        p = os.path.join(src_dir, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(data, fn))
            print(f"    {fn:16s} {human(os.path.getsize(p))}")

    # видео: готовый HLS → своё пережатие → исходный файл. Плеер понимает и то и другое
    hls_src = os.path.join(src_dir, "hls")
    video = os.path.join(src_dir, "video.mp4")
    if os.path.isdir(hls_src):
        shutil.copytree(hls_src, os.path.join(data, "hls"))
        print("    hls/            (готовый, из прогона)")
    elif args.hls and os.path.exists(video):
        print(f"    пережимаю в HLS, 720p {args.bitrate} (~20× реалтайма)…", flush=True)
        make_hls(os.path.realpath(video), data, args.bitrate)
        seg = dir_size(os.path.join(data, "hls"))
        was = os.path.getsize(os.path.realpath(video))
        print(f"    hls/            {human(seg)} вместо {human(was)}")
    elif os.path.exists(video):
        shutil.copy(video, os.path.join(data, "video.mp4"))
        print(f"    video.mp4       {human(os.path.getsize(video))}")

    # в архиве нечем сохранять имена спикеров — прячем кнопку
    meta["viewonly"] = True
    with open(os.path.join(data, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    shutil.copytree(os.path.join(ROOT, "player"), os.path.join(pkg, "player"))
    shutil.copy(os.path.join(ROOT, "scripts", "serve.py"), os.path.join(pkg, "serve.py"))

    url = f"/player/?src=/videro-out/{args.name}/"
    for fn, body in (("start-mac.command", LAUNCH_SH), ("start-linux.sh", LAUNCH_SH),
                     ("start-windows.bat", LAUNCH_BAT)):
        p = os.path.join(pkg, fn)
        with open(p, "w", encoding="utf-8", newline="\r\n" if fn.endswith(".bat") else "\n") as f:
            f.write(body.format(url=url))
        if not fn.endswith(".bat"):
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    dur = tl.get("duration_sec", 0)
    speakers = {s["speaker"] for s in tl.get("transcript") or [] if s.get("speaker")}
    stats = (f"  видео           {int(dur // 3600)}:{int(dur % 3600 // 60):02d}:{int(dur % 60):02d}\n"
             f"  глав            {len(tl.get('chapters') or [])}\n"
             f"  хайлайтов       {len(tl.get('highlights') or [])}\n"
             f"  реплик          {len(tl.get('transcript') or [])}\n"
             f"  спикеров        {len(speakers)}")
    with open(os.path.join(pkg, "README.txt"), "w", encoding="utf-8") as f:
        f.write(READ_ME.format(title=meta.get("title") or args.name, stats=stats))

    total = sum(os.path.getsize(os.path.join(d, x))
                for d, _, fs in os.walk(pkg) for x in fs)
    print(f"  папка готова: {human(total)}")

    if args.no_zip:
        print(f"\n✔ {pkg}")
        return

    zpath = pkg + ".zip"
    print("  жму в zip (видео не сжимается, только упаковывается)…", flush=True)
    with zipfile.ZipFile(zpath, "w", allowZip64=True) as z:
        for d, _, fs in os.walk(pkg):
            for x in fs:
                full = os.path.join(d, x)
                rel = os.path.join(args.name + "-share", os.path.relpath(full, pkg))
                # медиа уже сжато, второй проход дал бы проценты за минуты работы
                mode = (zipfile.ZIP_STORED if os.path.splitext(x)[1].lower() in MEDIA_EXT
                        else zipfile.ZIP_DEFLATED)
                z.write(full, rel, compress_type=mode)
    shutil.rmtree(pkg, ignore_errors=True)
    print(f"\n✔ {zpath}  ({human(os.path.getsize(zpath))})")
    print("  отдавай этот файл — коллеге нужен только Python 3")


if __name__ == "__main__":
    main()
