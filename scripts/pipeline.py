#!/usr/bin/env python3
"""Весь путь от видеофайла до открытого плеера одной командой.

  ./run.sh all путь/к/видео.mp4
  ./run.sh all видео.mp4 --name лекция-2 --title "Лекция 2" --lang ru --serve

Порядок шагов не случайный:
  разметка → нормализация терминов → диаризация → главы
Нормализация до глав — чтобы модель видела термины в правильном написании.
Диаризация до глав — chapters.py ставит границы по смене докладчика.

Каждый шаг пропускается, если его результат уже в timeline.json: состояние
берётся из самих данных, а не из файлов-меток, поэтому прогон можно прервать
и запустить заново — доделает с места обрыва. --force переделывает всё.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
PY = sys.executable


def run(script: str, *args: str) -> None:
    cmd = [PY, os.path.join(HERE, script), *args]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"\n✗ {script} завершился с кодом {r.returncode} — дальше не иду")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def state(tl_path: str) -> dict:
    """Что уже сделано — видно по самому timeline.json."""
    if not os.path.isfile(tl_path):
        return {"analyzed": False, "normalized": False, "diarized": False, "chaptered": False}
    tl = load(tl_path)
    tr = tl.get("transcript") or []
    return {
        "analyzed": bool(tl.get("scenes")),
        "normalized": any("text_asr" in s for s in tr),
        "diarized": any(s.get("speaker") for s in tr),
        "chaptered": bool(tl.get("chapters")),
        "n_scenes": len(tl.get("scenes") or []),
        "n_segments": len(tr),
    }


def step(name: str, done: bool, force: bool) -> bool:
    mark = "уже есть, пропускаю" if done and not force else "делаю"
    print(f"\n{'─' * 62}\n▶ {name} — {mark}\n{'─' * 62}", flush=True)
    return force or not done


def main():
    ap = argparse.ArgumentParser(description="полный прогон: видео → плеер")
    ap.add_argument("video", help="путь к видеофайлу")
    ap.add_argument("--name", default=None, help="имя папки в videro-out (по умолчанию имя файла)")
    ap.add_argument("--title", default=None, help="заголовок на странице плеера")
    ap.add_argument("--lang", default="ru", help="язык речи (default ru)")
    ap.add_argument("--hls", action="store_true",
                    help="сделать HLS: примерно вдвое легче исходника и сегментами "
                         "по паре МБ; по умолчанию играем исходный файл как есть")
    ap.add_argument("--bitrate", default="350k", help="битрейт 720p при --hls (default 350k)")
    ap.add_argument("--serve", action="store_true", help="поднять плеер в конце")
    ap.add_argument("--port", default="8777")
    ap.add_argument("--force", action="store_true", help="переделать все шаги заново")
    args = ap.parse_args()

    src = os.path.abspath(args.video)
    if not os.path.isfile(src):
        sys.exit(f"файл не найден: {src}")
    name = args.name or os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(ROOT, "videro-out", name)
    tl_path = os.path.join(out, "timeline.json")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    st = state(tl_path)
    print(f"▶ {os.path.basename(src)}\n  → videro-out/{name}")
    if st["analyzed"]:
        print(f"  найден прогон: {st['n_scenes']} сцен, {st['n_segments']} реплик")
        print("  сделано: " + ", ".join(k for k in
              ("normalized", "diarized", "chaptered") if st[k]) or "  сделано: только разметка")

    # ── видео и обложка для плеера ───────────────────────────────────────────
    link = os.path.join(out, "video.mp4")
    if not args.hls and not os.path.exists(link):
        os.symlink(src, link)                      # без копии на 600 МБ
    poster = os.path.join(out, "poster.jpg")
    if not os.path.exists(poster):                 # --no-hls пропускает make_poster апстрима
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "120", "-i", src,
                        "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3", poster],
                       check=False)
        if not os.path.exists(poster):             # видео короче двух минут
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", src,
                            "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3", poster],
                           check=False)
    meta = os.path.join(out, "meta.json")
    if not os.path.exists(meta):
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"title": args.title or name, "description": ""}, f, ensure_ascii=False, indent=2)

    # ── шаги ─────────────────────────────────────────────────────────────────
    if step("разметка: whisper + сцены (самый долгий шаг)", st["analyzed"], args.force):
        # всегда --no-hls: лесенка апстрима на записи экрана даёт файл тяжелее
        # исходника, свой HLS делаем ниже одной дорожкой
        run("videro_local.py", src, "--lang", args.lang, "--out", out, "--no-hls")

    if step("нормализация терминов по glossary.txt", st["normalized"], args.force):
        run("normalize.py", tl_path)

    if step("диаризация: кто когда говорит", st["diarized"], args.force):
        run("diarize.py", src, "--timeline", tl_path)

    if step("главы и хайлайты", st["chaptered"], args.force):
        run("chapters.py", tl_path)

    if args.hls and not os.path.isdir(os.path.join(out, "hls")):
        print(f"\n{'─' * 62}\n▶ HLS, 720p {args.bitrate}\n{'─' * 62}", flush=True)
        from hlsmake import dir_size, make_hls
        make_hls(os.path.realpath(link if os.path.exists(link) else src), out, args.bitrate)
        print(f"  {dir_size(os.path.join(out, 'hls')) / 1024 / 1024:.0f} МБ "
              f"вместо {os.path.getsize(src) / 1024 / 1024:.0f} МБ")

    print(f"\n{'─' * 62}\n▶ проверка на секреты\n{'─' * 62}", flush=True)
    run("redact.py", tl_path, "--ocr", "--dry-run")

    fin = state(tl_path)
    tl = load(tl_path)
    print(f"\n{'═' * 62}")
    print(f"✔ готово за {(time.time() - t0) / 60:.0f} мин: {fin['n_scenes']} сцен, "
          f"{len(tl.get('chapters') or [])} глав, {len(tl.get('highlights') or [])} хайлайтов, "
          f"{fin['n_segments']} реплик, "
          f"{len({s['speaker'] for s in tl.get('transcript') or [] if s.get('speaker')})} спикеров")
    url = f"http://localhost:{args.port}/player/?src=/videro-out/{name}/"
    print(f"\n  плеер: {url}")
    print("  дальше: открой плеер, нажми «Назвать спикеров» и впиши имена")
    print(f"{'═' * 62}")

    if args.serve:
        print("\n(Ctrl+C чтобы остановить сервер)\n")
        os.execv(PY, [PY, os.path.join(HERE, "serve.py"), ROOT, "--port", args.port])
    else:
        print(f"\n  поднять плеер: ./run.sh serve videro-out/{name} {args.port}")


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        sys.exit("нет ffmpeg")
    main()
